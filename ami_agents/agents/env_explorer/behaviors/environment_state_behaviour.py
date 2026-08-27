import json
from typing import Dict, Any, Optional
from spade.behaviour import CyclicBehaviour

from ....shared.models.messages import MessageType, META_CORRELATION_ID
from ....shared.utils.demo_log import demo


class EnvironmentStateBehaviour(CyclicBehaviour):
    """
    Handles ENV_STATE_REQUEST messages and provides current state information
    about artifacts, including full snapshots or filtered property queries.
    """

    async def run(self):
        """Main behavior loop - handles environment state requests."""
        msg = await self.receive(timeout=1)
        if not msg:
            return

        # Only process ENV_STATE_REQUEST messages
        msg_type = msg.get_metadata("type")
        if msg_type != MessageType.ENV_STATE_REQUEST.value:
            return

        self.agent.logger.info(demo(f"Received ENV_STATE_REQUEST from {msg.sender}"))

        # Parse request payload
        try:
            payload = json.loads(msg.body or "{}")
        except json.JSONDecodeError:
            payload = {}
            self.agent.logger.warning("Failed to parse state request payload, using empty payload")

        # Extract request parameters from LLM extraction
        artifact_name = payload.get("artifact_name")
        artifact_type = payload.get("artifact_type")
        workspace_type = payload.get("workspace_type")
        property_name = payload.get("property_name")
        parameter_name = payload.get("parameter_name")

        self.agent.logger.info(demo(
            f"Processing state request: artifact_name={artifact_name}, property_name={property_name}, "
            f"artifact_type={artifact_type}, workspace_type={workspace_type}, parameter_name={parameter_name}"
        ))

        # Resolve all constraints in one query (SPARQL-like with optional selectors)
        response_payload = self._query_state(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            workspace_type=workspace_type,
            property_name=property_name,
            parameter_name=parameter_name
        )

        # Send response
        reply = msg.make_reply()
        reply.body = json.dumps(response_payload)
        reply.set_metadata("type", MessageType.ENV_STATE_RESPONSE.value)

        # Preserve correlation ID and thread
        correlation_id = msg.get_metadata("correlation_id")
        if correlation_id:
            reply.set_metadata("correlation_id", correlation_id)
        if msg.thread:
            reply.thread = msg.thread

        await self.send(reply)

        # Log response summary
        if "error" in response_payload:
            self.agent.logger.info(demo(f"Sent error response: {response_payload.get('error')}"))
        else:
            self.agent.logger.info(demo(f"Sent state response"))

    def _generate_state_response(self, artifact_id: Optional[str], property_uri: Optional[str]) -> Dict[str, Any]:
        """
        Generate environment state response based on request parameters.

        Args:
            artifact_id: Specific artifact to query (None for full snapshot)
            property_uri: Specific property to query (None for all properties)

        Returns:
            Dictionary containing state information or error details
        """
        self.agent.logger.info(demo(f"Generating state response for artifact_id={artifact_id}, property_uri={property_uri}"))
        try:
            if artifact_id:
                return self._get_artifact_state(artifact_id, property_uri)
            else:
                return self._get_full_environment_snapshot()
        except Exception as e:
            self.agent.logger.error(f"Error generating state response: {e}", exc_info=True)
            return {
                "error": "state_generation_failed",
                "detail": str(e)
            }

    def _get_artifact_state(self, artifact_id: str, property_uri: Optional[str]) -> Dict[str, Any]:
        """
        Get state information for a specific artifact.

        Args:
            artifact_id: The artifact identifier
            property_uri: Optional specific property to query

        Returns:
            Dictionary containing artifact state or error information
        """
        # Find the artifact
        artifact = self.agent.artifacts.get(artifact_id)
        if not artifact:
            return {
                "error": "artifact_not_found",
                "artifact_id": artifact_id,
            }

        # Get the current state
        state = dict(artifact.current_state)

        if property_uri:
            # Return specific property
            if property_uri in state:
                return {
                    "artifact_id": artifact_id,
                    "property_uri": property_uri,
                    "value": state.get(property_uri),
                }
            else:
                return {
                    "error": "property_not_found",
                    "artifact_id": artifact_id,
                    "property_uri": property_uri,
                    "available_properties": list(state.keys())
                }
        else:
            # Return full artifact state
            return {
                "artifact_id": artifact_id,
                "name": artifact.name,
                "workspace_id": getattr(artifact, "workspace_id", None),
                "state": state,
                "property_count": len(state)
            }

    def _get_full_environment_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of all artifacts in the environment.

        Returns:
            Dictionary containing all artifacts and their states
        """
        artifacts_snapshot = {}
        total_properties = 0

        for aid, artifact in self.agent.artifacts.items():
            artifact_state = dict(artifact.current_state)
            artifacts_snapshot[aid] = {
                "name": artifact.name,
                "workspace_id": getattr(artifact, "workspace_id", None),
                "state": artifact_state,
            }
            total_properties += len(artifact_state)

        return {
            "artifacts": artifacts_snapshot,
            "summary": {
                "total_artifacts": len(artifacts_snapshot),
                "total_properties": total_properties,
                "workspaces": list(set(
                    artifact.get("workspace_id") for artifact in artifacts_snapshot.values()
                    if artifact.get("workspace_id")
                ))
            }
        }

    def _query_state(
        self,
        artifact_name: Optional[str] = None,
        artifact_type: Optional[str] = None,
        workspace_type: Optional[str] = None,
        property_name: Optional[str] = None,
        parameter_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query artifact state with optional constraints (SPARQL-like).

        All parameters are optional and "NA" is treated as unconstrained.
        Returns all results matching ALL non-NA constraints.

        If parameter_name is specified without property_name, it still constrains:
        the query searches for any property whose output_schema has that parameter.

        Args:
            artifact_name: Artifact name (e.g., "light308") or "NA"
            artifact_type: Artifact semantic type (e.g., "ex:Light") or "NA"
            workspace_type: Workspace semantic type (e.g., "ex:Kitchen") or "NA"
            property_name: Property name (e.g., "brightness") or "NA"
            parameter_name: Parameter within property output (e.g., "color") or "NA"

        Returns:
            Single result dict if one match, {"results": [..]} if multiple, or error dict
        """
        results = []

        # Filter artifacts by constraints
        for aid, artifact in self.agent.artifacts.items():
            self.agent.logger.info(demo(f"[_query_state] Checking artifact {aid} ({artifact.name})"))

            # Check artifact_name constraint
            if artifact_name and artifact_name != "NA":
                if artifact.name != artifact_name:
                    self.agent.logger.info(demo(f"  ✗ Name mismatch: {artifact.name} != {artifact_name}"))
                    continue
                self.agent.logger.info(demo(f"  ✓ Name matches: {artifact.name}"))

            # Check artifact_type constraint (match ex: semantic types only)
            if artifact_type and artifact_type != "NA":
                artifact_semantic_types = getattr(artifact, "semantic_types", []) or []
                # Filter to ex: types only
                ex_types = [st for st in artifact_semantic_types if isinstance(st, str) and st.startswith("ex:")]
                if artifact_type not in ex_types:
                    self.agent.logger.debug(f"  ✗ Type mismatch: {artifact_type} not in {ex_types}")
                    continue
                self.agent.logger.debug(f"  ✓ Type matches: {artifact_type}")

            # Check workspace_type constraint (match ex: semantic types only)
            if workspace_type and workspace_type != "NA":
                workspace_id = getattr(artifact, "workspace_id", None)
                workspace_matches = False
                if workspace_id:
                    ws = self.agent.environment_map.get(workspace_id)
                    if ws:
                        ws_semantic_types = getattr(ws, "semantic_types", []) or []
                        # Filter to ex: types only
                        ex_types = [st for st in ws_semantic_types if isinstance(st, str) and st.startswith("ex:")]
                        if workspace_type in ex_types:
                            workspace_matches = True
                if not workspace_matches:
                    continue

            # Check if property/parameter constraints apply
            has_property_constraint = property_name and property_name != "NA"
            has_parameter_constraint = parameter_name and parameter_name != "NA"

            if has_property_constraint or has_parameter_constraint:
                # Find matching properties
                affs = self.agent.integration_engine.get_affordances_for_artifact(aid)
                for aff in affs:
                    if aff.affordance_type.value != "property":
                        continue

                    # Check property_name constraint
                    if has_property_constraint and aff.name != property_name:
                        continue

                    # Check parameter_name constraint (must have this param in output_schema)
                    if has_parameter_constraint:
                        output_schema = aff.output_schema or {}
                        schema_props = output_schema.get("properties", {})
                        if parameter_name not in schema_props:
                            continue

                    # Found matching property
                    if aff.form and aff.form.href:
                        state_data = self._generate_state_response(aid, aff.form.href)
                        results.append(state_data)
            else:
                # No property/parameter constraints; return full artifact state
                state_data = self._generate_state_response(aid, None)
                results.append(state_data)

        ## log the number and details of results for debugging (can be removed in production)
        self.agent.logger.debug(f"State query found {len(results)} matching artifacts")
        for r in results:
            self.agent.logger.debug(f"Matching artifact state: {r}")

        # Return results
        if not results:
            return {
                "error": "no_matching_results",
                "constraints": {
                    "artifact_name": artifact_name,
                    "artifact_type": artifact_type,
                    "workspace_type": workspace_type,
                    "property_name": property_name,
                    "parameter_name": parameter_name,
                }
            }

        # Return single result directly, multiple results wrapped
        return results[0] if len(results) == 1 else {"results": results}

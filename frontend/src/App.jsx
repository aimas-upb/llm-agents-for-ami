import React, { useState, useEffect, useCallback } from 'react';
import * as rdflib from 'rdflib';
import config from '../../conf/cartago_config_light308.json';
import WorkspaceTree from './components/WorkspaceTree';
import ArtifactList from './components/ArtifactList';
import ArtifactDetails from './components/ArtifactDetails';

const JS = rdflib.Namespace('https://www.w3.org/2019/wot/json-schema#');

function App() {
  const [tree, setTree] = useState([]);
  const [navHistory, setNavHistory] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [selectedArtifact, setSelectedArtifact] = useState(null);
  const [artifactDetails, setArtifactDetails] = useState(null);
  const [actionPayloads, setActionPayloads] = useState({});

  // Build workspace tree from config
  useEffect(() => {
    const workspaces = config["environment-config"]?.workspaces || [];
    const workspaceMap = {};
    const roots = [];

    workspaces.forEach(ws => {
      workspaceMap[ws.name] = { ...ws, children: [] };
    });

    workspaces.forEach(ws => {
      if (ws["parent-name"]) {
        const parent = workspaceMap[ws["parent-name"]];
        if (parent) {
          parent.children.push(workspaceMap[ws.name]);
        }
      } else {
        roots.push(workspaceMap[ws.name]);
      }
    });

    setTree(roots);
  }, []);

  // Determine current nodes based on navigation history
  const currentNodes = navHistory.length === 0 ? tree : navHistory[navHistory.length - 1].children || [];

  const handleNodeClick = (node) => {
    setNavHistory([...navHistory, node]);
    setSelectedArtifact(null);
    setArtifactDetails(null);
    setArtifacts([]);
  };

  const handleBack = () => {
    if (navHistory.length > 0) {
      const newNavHistory = [...navHistory];
      newNavHistory.pop();
      setNavHistory(newNavHistory);
      setSelectedArtifact(null);
      setArtifactDetails(null);
      setArtifacts([]);
    }
  };

  // Fetch artifacts for the current workspace
  useEffect(() => {
    if (navHistory.length > 0) {
      const currentWorkspace = navHistory[navHistory.length - 1];
      const url = `http://localhost:8080/workspaces/${currentWorkspace.name}`;

      fetch(url)
        .then(response => response.text())
        .then(turtleData => {
          const store = rdflib.graph();
          rdflib.parse(turtleData, store, url, 'text/turtle', (err) => {
            if (err) {
              console.error("Error parsing turtle data:", err);
              return;
            }

            const HMAS = rdflib.Namespace('https://purl.org/hmas/');
            const JACAMO = rdflib.Namespace('https://purl.org/hmas/jacamo/');
            const RDF = rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#');

            const currentSubject = `${url}#workspace`;
            const containsStatements = store.statementsMatching(
              rdflib.sym(currentSubject),
              HMAS('contains'),
              undefined
            );

            const artifactURIs = [];
            containsStatements.forEach(stmt => {
              const candidateObj = stmt.object;
              const isArtifact = store.statementsMatching(
                candidateObj,
                RDF('type'),
                HMAS('Artifact')
              ).length > 0;
              const isBody = store.statementsMatching(
                candidateObj,
                RDF('type'),
                JACAMO('Body')
              ).length > 0;

              if (isArtifact && !isBody) {
                artifactURIs.push(candidateObj.value);
              }
            });

            const artifactsList = artifactURIs.map(uri => uri.split('/').pop().split('#')[0]);
            setArtifacts(artifactsList);
          });
        })
        .catch(console.error);
    }
  }, [navHistory]);

  // Fetch artifact details when one is selected
  useEffect(() => {
    if (selectedArtifact) {
      fetch(selectedArtifact.url)
        .then(response => response.text())
        .then(turtleData => {
          const store = rdflib.graph();
          rdflib.parse(turtleData, store, selectedArtifact.url, 'text/turtle', (err) => {
            if (err) {
              console.error("Error parsing artifact turtle data:", err);
              return;
            }

            const TD = rdflib.Namespace('https://www.w3.org/2019/wot/td#');
            const HTV = rdflib.Namespace('http://www.w3.org/2011/http#');
            const HCTL = rdflib.Namespace('https://www.w3.org/2019/wot/hypermedia#');
            const RDF = rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#');

            const artifactSubject = rdflib.sym(selectedArtifact.url + "#artifact");
            const properties = [];
            const actionStatements = store.match(artifactSubject, TD('hasActionAffordance'), undefined);
            
            const defaultActions = new Set([
              'getArtifactRepresentation',
              'updateArtifactRepresentation',
              'deleteArtifactRepresentation',
              'focusArtifact',
              'subscribeToArtifact',
              'unsubscribeFromArtifact'
            ]);

            const actions = actionStatements.map(stmt => {
              const actionNode = stmt.object;
              const inputSchemaStmt = store.any(actionNode, TD('hasInputSchema'));
              
              return {
                name: store.any(actionNode, TD('name'))?.value,
                title: store.any(actionNode, TD('title'))?.value,
                inputSchemaNode: inputSchemaStmt,
                forms: store.match(actionNode, TD('hasForm'), undefined).map(formStmt => ({
                  method: store.any(formStmt.object, HTV('methodName'))?.value,
                  target: store.any(formStmt.object, HCTL('hasTarget'))?.value,
                  contentType: store.any(formStmt.object, HCTL('forContentType'))?.value
                }))
              };
            }).filter(action => !defaultActions.has(action.name));

            const eventStatements = store.match(artifactSubject, TD('hasEventAffordance'), undefined);
            const events = eventStatements.map(stmt => {
              const eventNode = stmt.object;
              return {
                name: store.any(eventNode, TD('name'))?.value,
                title: store.any(eventNode, TD('title'))?.value
              };
            });

            store.match(artifactSubject, undefined, undefined).forEach(stmt => {
              if (![
                TD('hasActionAffordance').value,
                TD('hasEventAffordance').value
              ].includes(stmt.predicate.value)) {
                properties.push({
                  predicate: stmt.predicate.value,
                  object: stmt.object.value
                });
              }
            });

            setArtifactDetails({ properties, actions, events, store });
          });
        })
        .catch(console.error);
    }
  }, [selectedArtifact]);

  const handleArtifactClick = (artifactName) => {
    const currentWorkspace = navHistory[navHistory.length - 1];
    const artifactUrl = `http://localhost:8080/workspaces/${currentWorkspace.name}/artifacts/${artifactName}`;
    setSelectedArtifact({ name: artifactName, url: artifactUrl });
  };

  const handlePayloadChange = useCallback((actionName, data) => {
    setActionPayloads(prev => ({
      ...prev,
      [actionName]: data
    }));
  }, []);

  const handleExecuteAction = async (action) => {
    const form = action.forms?.[0];
    if (!form) return;
  
    try {
      const actionPayload = actionPayloads[action.name] || {};
      const { payload, payloadType } = actionPayload;
      let jsonBody = {};
  
      // Check for required properties in the input schema
      const inputSchema = action.inputSchemaNode;
      const hasRequiredProperties = inputSchema && artifactDetails?.store.match(
        inputSchema,
        JS('required'),
        undefined
      ).length > 0;
  
      // Handle RDF payload
      if (payloadType === 'rdf' && payload) {
        const rdfStore = rdflib.graph();
        const baseURI = 'http://example.com/';
        await new Promise((resolve, reject) => {
          rdflib.parse(payload, rdfStore, baseURI, 'text/turtle', (err) => {
            if (err) reject(err);
            else resolve();
          });
        });
  
        // Find the first subject with properties
        const subject = rdfStore.match(undefined, undefined, undefined)
          .find(s => rdfStore.match(s.subject, undefined, undefined).length > 0)?.subject;
  
        if (subject) {
          rdfStore.match(subject, undefined, undefined).forEach(predicateStmt => {
            const predicate = predicateStmt.predicate.value.split(/[#\/]/).pop();
            const obj = predicateStmt.object;
            
            if (obj.termType === 'Literal') {
              jsonBody[predicate] = obj.value;
              if (obj.datatype && typeof obj.datatype.value === 'string') {
                if (obj.datatype.value.endsWith('#decimal') || obj.datatype.value.endsWith('#integer')) {
                  jsonBody[predicate] = parseFloat(obj.value);
                } else if (obj.datatype.value.endsWith('#boolean')) {
                  jsonBody[predicate] = obj.value.toLowerCase() === 'true';
                }
              }
            } else {
              jsonBody[predicate] = obj.value;
            }
          });
        }
      }
      // Handle JSON payload
      else if (payloadType === 'json' && payload) {
        try {
          jsonBody = JSON.parse(payload);
        } catch (e) {
          if (payload.includes(":")) {
            const [key, val] = payload.split(":")
              .map(s => s.trim().replace(/^"+|"+$/g, ''));
            jsonBody = { [key]: val };
          }
        }
      }
  
      // Validate required properties if needed
      if (hasRequiredProperties && inputSchema) {
        const requiredProps = artifactDetails.store.match(inputSchema, JS('required'), undefined)
          .map(req => req.object.value);
  
        const missingProps = requiredProps.filter(prop => !(prop in jsonBody));
        if (missingProps.length > 0) {
          throw new Error(`Missing required properties: ${missingProps.join(', ')}`);
        }
        
        if (Object.keys(jsonBody).length === 0) {
          throw new Error('Payload is required for this action');
        }
      }
  
      const options = {
        method: form.method,
        headers: {
          'Content-Type': 'application/json',
          'X-Agent-WebID': 'http://localhost:8080/agents/alex'
        }
      };
      
      if (hasRequiredProperties || Object.keys(jsonBody).length > 0) {
        options.body = JSON.stringify(jsonBody);
      }
  
      const response = await fetch(form.target, options);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      const result = await response.text();
      alert(`Action succeeded: ${result}`);
    } catch (error) {
      console.error('Action failed:', error);
      alert(`Action failed: ${error.message}`);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Workspace Hierarchy</h1>
      <WorkspaceTree 
        navHistory={navHistory} 
        currentNodes={currentNodes} 
        handleNodeClick={handleNodeClick} 
        handleBack={handleBack}
      />
      <ArtifactList 
        artifacts={artifacts} 
        handleArtifactClick={handleArtifactClick} 
      />
      {selectedArtifact && artifactDetails && (
        <ArtifactDetails 
          selectedArtifact={selectedArtifact} 
          artifactDetails={artifactDetails} 
          handleExecuteAction={handleExecuteAction}
          handlePayloadChange={handlePayloadChange}
          onBack={() => setSelectedArtifact(null)}
        />
      )}
    </div>
  );
}

export default App;

import os
import sys
import uvicorn
import httpx
import logging
import urllib.parse
import argparse
from fastapi import FastAPI, Request, HTTPException, Response

# Configuration
# The address where this subscriber receives callbacks
SUBSCRIBER_HOST = os.getenv("SUBSCRIBER_HOST", "localhost")
SUBSCRIBER_PORT = int(os.getenv("SUBSCRIBER_PORT", "8085"))
SUBSCRIBER_CALLBACK_PATH = "/callback"
SUBSCRIBER_URL = f"http://{SUBSCRIBER_HOST}:{SUBSCRIBER_PORT}{SUBSCRIBER_CALLBACK_PATH}"

# The Adapter URL (e.g., http://localhost:8080)
ADAPTER_URL = os.getenv("ADAPTER_URL", "http://localhost:8080").rstrip("/")
HUB_URL = f"{ADAPTER_URL}/hub/"

# Parse command line arguments
parser = argparse.ArgumentParser(description="WebSub Subscriber for Yggdrasil HA Adapter.")
parser.add_argument("workspace_name", type=str,
                    help="The name of the workspace to subscribe to (e.g., lab308).")
parser.add_argument("--scope", type=str, choices=["artifact", "workspace"], default="workspace",
                    help="Scope of subscription: 'artifact' (default) or 'workspace'.")
args = parser.parse_args()

WORKSPACE_NAME = args.workspace_name
# For simplicity, we'll assume a default artifact name.
# In a real scenario, you might discover artifacts or specify them.
DEFAULT_ARTIFACT_NAME = "lights_308" 

if args.scope == "workspace":
    TOPIC_URL = f"{ADAPTER_URL}/workspaces/{WORKSPACE_NAME}"
else:
    TOPIC_URL = f"{ADAPTER_URL}/workspaces/{WORKSPACE_NAME}/artifacts/{DEFAULT_ARTIFACT_NAME}#artifact"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("WebSubSubscriber")

app = FastAPI()

@app.get(SUBSCRIBER_CALLBACK_PATH)
async def handle_verification(request: Request):
    """
    Handles WebSub Intent Verification.
    The hub sends a GET request with hub.challenge, hub.topic, and hub.mode.
    We must echo back the hub.challenge.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    topic = params.get("hub.topic")
    challenge = params.get("hub.challenge")

    if mode and topic and challenge:
        logger.info(f"Received verification challenge for mode '{mode}' on topic '{topic}'")
        # In a real app, verify the topic is one we requested
        return Response(content=challenge, media_type="text/plain", status_code=200)
    
    logger.warning("Received invalid verification request")
    raise HTTPException(status_code=400, detail="Invalid verification request")

@app.post(SUBSCRIBER_CALLBACK_PATH)
async def handle_notification(request: Request):
    """
    Handles WebSub Content Distribution.
    The hub sends a POST request with the new content (event payload).
    """
    try:
        payload = await request.json()
        
        # Extract key information for the summary
        artifact = payload.get("artifactUri", "unknown")
        prop = payload.get("propertyUri", "unknown")
        val = payload.get("value")
        
        # Determine a cleaner name for the artifact if possible
        artifact_name_from_payload = artifact.split("/")[-1].split("#")[0]
        artifact_name_from_payload = urllib.parse.unquote(artifact_name_from_payload)
        
        # Determine a cleaner name for the property
        prop_name = prop.split("/")[-1]

        logger.info(f"EVENT RECEIVED >> Artifact: {artifact_name_from_payload} | Property: {prop_name} | Value: {val}")
        
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Failed to process notification: {e}")
        return Response(status_code=500)

async def subscribe():
    """Sends the subscription request to the Hub."""
    logger.info(f"Subscribing to {TOPIC_URL} at hub {HUB_URL} for workspace {WORKSPACE_NAME}...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                HUB_URL,
                data={
                    "hub.mode": "subscribe",
                    "hub.topic": TOPIC_URL,
                    "hub.callback": SUBSCRIBER_URL,
                    "hub.lease_seconds": 3600
                }
            )
            if response.status_code == 202:
                logger.info("Subscription request accepted by Hub.")
            else:
                logger.error(f"Subscription failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Could not connect to Hub: {e}")

@app.on_event("startup")
async def startup_event():
    # Schedule the subscription to run slightly after startup so the server is ready to receive the verification
    import asyncio
    asyncio.create_task(subscribe())

if __name__ == "__main__":
    print(f"Starting WebSub Subscriber on {SUBSCRIBER_HOST}:{SUBSCRIBER_PORT}")
    print(f"Callback URL: {SUBSCRIBER_URL}")
    print(f"Target Adapter URL: {ADAPTER_URL}")
    print(f"Target Workspace: {WORKSPACE_NAME}")
    print(f"Subscribing to Topic: {TOPIC_URL}")
    
    uvicorn.run(app, host=SUBSCRIBER_HOST, port=SUBSCRIBER_PORT)

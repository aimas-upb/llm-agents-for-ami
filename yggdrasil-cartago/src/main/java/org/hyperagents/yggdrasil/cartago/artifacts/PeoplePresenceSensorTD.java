package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.schemas.BooleanSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;

import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.CompletableFuture;

import javax.json.Json;
import javax.json.JsonObject;
import io.vertx.core.Vertx;
import io.vertx.core.Future;
import io.vertx.core.Promise;
import io.vertx.core.buffer.Buffer;
import io.vertx.ext.web.client.HttpResponse;
import io.vertx.ext.web.client.WebClient;

import org.hyperagents.yggdrasil.eventbus.messageboxes.HttpNotificationDispatcherMessagebox;
import org.hyperagents.yggdrasil.eventbus.messages.HttpNotificationDispatcherMessage;
import org.hyperagents.yggdrasil.utils.HttpInterfaceConfig;
import org.hyperagents.yggdrasil.utils.WebSubConfig;

/**
 * PeoplePresenceSensor TD Artifact, detects human presence.
 * Serves as an exemplary Hypermedia Artifact using the TD ontology.
 * Presence is checked periodically, and notifications are sent on updates.
 */
public class PeoplePresenceSensorTD extends HypermediaTDArtifact implements VertxInjectable, NeedsPeriodic {
  private Vertx vertxInstance;
  private HttpInterfaceConfig httpConfig;
  private WebSubConfig webSubConfig;
  private HttpNotificationDispatcherMessagebox dispatcherMessagebox;
  private static final String DEFAULT_CONFIG_VALUE = "default";
  private static final long PRESENCE_CHECK_INTERVAL = 1000;

  public void init() {
    this.defineObsProperty("presence", false);
  }
  
  public void init(final boolean presence) {
    this.defineObsProperty("presence", presence);
  }

  @Override
  public void setVertx(Vertx vertx) {
    this.vertxInstance = vertx;
    if (this.vertxInstance != null) {
        this.httpConfig = this.vertxInstance
            .sharedData()
            .<String, HttpInterfaceConfig>getLocalMap("http-config")
            .get(DEFAULT_CONFIG_VALUE);
        this.webSubConfig = this.vertxInstance
            .sharedData()
            .<String, WebSubConfig>getLocalMap("notification-config")
            .get(DEFAULT_CONFIG_VALUE);
        if (this.webSubConfig != null && this.vertxInstance.eventBus() != null) {
             this.dispatcherMessagebox = new HttpNotificationDispatcherMessagebox(
                this.vertxInstance.eventBus(),
                this.webSubConfig
            );
        } else {
            if (this.webSubConfig == null) {
              System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: WebSubConfig not found in shared data.");
            }
             // It's unlikely eventBus is null if vertxInstance is not, but good to be safe.
            if (this.vertxInstance.eventBus() == null) {
              System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: Vert.x EventBus is null.");
            }
        }
        if (this.httpConfig == null) {
             System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: HttpInterfaceConfig not found in shared data.");
        }
    }
  }

  @Override
  public void startPeriodicTasks() {
    if (this.vertxInstance != null && this.httpConfig != null && this.dispatcherMessagebox != null) {
        // Note: retrieveCurrentSimulationTime is a blocking call.
        // For high-frequency tasks or many such artifacts, consider making it async.
        vertxInstance.setPeriodic(PRESENCE_CHECK_INTERVAL, timerId -> {
            retrieveCurrentSimulationTimeAsync()
                .onSuccess(currentTime -> {
                    try {
                        double hour = currentTime / 60.0;
                        boolean newPresenceValue = false;

                        if (hour >= 8 && hour <= 18) {
                            newPresenceValue = Math.random() < 0.9;
                        }

                        boolean currentPresenceValue = (Boolean) this.getObsProperty("presence").getValue();

                        if (newPresenceValue != currentPresenceValue) {
                            this.getObsProperty("presence").updateValue(newPresenceValue);

                            String currentWorkspaceName = getId().getWorkspaceId().getName();
                            String currentArtifactName = getId().getName();
                            String artifactUri = this.httpConfig.getArtifactUri(currentWorkspaceName, currentArtifactName);
                            String propertyName = "presence";
                            String propertyUri = artifactUri + "/props/" + propertyName;
                            String triggerUri = propertyUri;

                            JsonObject payload = Json.createObjectBuilder()
                                .add("artifactUri", artifactUri)
                                .add("propertyUri", propertyUri)
                                .add("value", newPresenceValue)
                                .add("valueTypeUri", "http://www.w3.org/2001/XMLSchema#boolean")
                                .add("timestamp", DateTimeFormatter.ISO_INSTANT.format(Instant.now()))
                                .add("triggerUri", triggerUri)
                                .build();

                            String notificationTargetUri = this.httpConfig.getArtifactUriFocusing(currentWorkspaceName, currentArtifactName);

                            this.dispatcherMessagebox.sendMessage(
                                new HttpNotificationDispatcherMessage.ArtifactObsPropertyUpdated(
                                    notificationTargetUri,
                                    payload.toString()
                                )
                            );
                        }
                    } catch (Exception e) {
                         System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: Error processing presence after time retrieval: " + e.getMessage());
                    }
                })
                .onFailure(err -> {
                    System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: Error during periodic presence check (failed to retrieve time): " + err.getMessage());
                });
        });
    } else {
        String reason = "";
        if (this.vertxInstance == null) reason += "Vert.x instance not initialized. ";
        if (this.httpConfig == null) reason += "HttpInterfaceConfig not initialized. ";
        if (this.dispatcherMessagebox == null) reason += "DispatcherMessagebox not initialized. ";
        System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: Cannot start periodic tasks. " + reason.trim());
    }
  }

  private Future<Long> retrieveCurrentSimulationTimeAsync() {
    Promise<Long> promise = Promise.promise();
    String clockUrl = "http://localhost:8080/workspaces/lab308/artifacts/clock308/timeOfDay";
    WebClient client = WebClient.create(this.vertxInstance);
    io.vertx.core.json.JsonObject requestBody = new io.vertx.core.json.JsonObject();

    client.postAbs(clockUrl)
      .putHeader("Content-Type", "application/json")
      .putHeader("X-Agent-WebID", "http://localhost:8080/agents/alex")
      .sendJsonObject(requestBody)
      .onSuccess(response -> {
        try {
          io.vertx.core.json.JsonObject timeJson = response.bodyAsJsonObject();
          if (timeJson != null && timeJson.containsKey("timeOfDay")) {
            promise.complete(timeJson.getLong("timeOfDay"));
          } else {
            System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: Invalid or missing timeOfDay in response from clock: " + response.bodyAsString());
            promise.fail("Invalid response from clock service for timeOfDay");
          }
        } catch (Exception e) {
          System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: Error parsing timeOfDay response: " + e.getMessage());
          promise.fail(e);
        }
      })
      .onFailure(err -> {
        System.err.println("[" + getId().getName() + "] PeoplePresenceSensorTD: Failed to retrieve simulation time: " + err.getMessage());
        promise.fail(err);
      });
    return promise.future();
  }
  
  /**
   * Retrieves the current presence detection status.
   */
  @OPERATION
  public void getPresence(final OpFeedbackParam<String> jsonPresence) {
    boolean currentPresence = (Boolean) this.getObsProperty("presence").getValue();
    JsonObject json = Json.createObjectBuilder()
        .add("presence", currentPresence)
        .build();
    jsonPresence.set(json.toString());
  }
  
  @Override
  protected void registerInteractionAffordances() {
    this.setSecurityScheme(SecurityScheme.getNoSecurityScheme());
    
    this.registerActionAffordance(
      "http://example.org/PresenceCommand",
      "getPresence",
      "presence",
      null,
      new ObjectSchema.Builder()
          .addProperty("presence", new BooleanSchema.Builder().build())
          .addRequiredProperties("presence")
          .build()
    );
  }
}

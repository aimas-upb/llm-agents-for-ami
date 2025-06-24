package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.schemas.NumberSchema;
import ch.unisg.ics.interactions.wot.td.schemas.IntegerSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;

import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.CompletableFuture;

import javax.json.Json;
import javax.json.JsonObject;
import io.vertx.core.Vertx;
import io.vertx.core.Future;
import io.vertx.core.Promise;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.HttpResponse;
import io.vertx.core.buffer.Buffer;

// import java.util.concurrent.ExecutionException; // No longer explicitly used with future.join()

import org.hyperagents.yggdrasil.cartago.artifacts.NeedsPeriodic;
import org.hyperagents.yggdrasil.eventbus.messageboxes.HttpNotificationDispatcherMessagebox;
import org.hyperagents.yggdrasil.eventbus.messages.HttpNotificationDispatcherMessage;
import org.hyperagents.yggdrasil.utils.HttpInterfaceConfig;
import org.hyperagents.yggdrasil.utils.WebSubConfig;

/**
 * LightSensor TD Artifact, measures luminosity.
 * Serves as an exemplary Hypermedia Artifact using the TD ontology.
 * Luminosity is checked periodically, and notifications are sent on updates.
 */
public class LightSensorTD extends HypermediaTDArtifact implements VertxInjectable, NeedsPeriodic {
  private Vertx vertxInstance;
  private HttpInterfaceConfig httpConfig;
  private WebSubConfig webSubConfig;
  private HttpNotificationDispatcherMessagebox dispatcherMessagebox;
  private static final String DEFAULT_CONFIG_VALUE = "default";
  private static final long LUMINOSITY_CHECK_INTERVAL = 1000;
  private static final double MAX_LUMINOSITY_FROM_LIGHTS = 500.0;


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
              System.err.println("[" + getId().getName() + "] LightSensorTD: WebSubConfig not found in shared data.");
            }
            if (this.vertxInstance.eventBus() == null) {
              System.err.println("[" + getId().getName() + "] LightSensorTD: Vert.x EventBus is null.");
            }
        }
        if (this.httpConfig == null) {
             System.err.println("[" + getId().getName() + "] LightSensorTD: HttpInterfaceConfig not found in shared data.");
        }
    }
  }
  
  public void init() {
    this.defineObsProperty("luminosity", 0.0);
  }
  
  public void init(final double luminosity) {
    this.defineObsProperty("luminosity", luminosity);
  }

  private double computeSunlightIntensity(double hour) {
    double intensity = 1000 * Math.sin(((hour % 24) - 6) * Math.PI / 12);
    return intensity < 0 ? 0 : intensity;
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
            System.err.println("[" + getId().getName() + "] LightSensorTD: Invalid or missing timeOfDay in response from clock: " + response.bodyAsString());
            promise.fail("Invalid response from clock service for timeOfDay");
          }
        } catch (Exception e) {
          System.err.println("[" + getId().getName() + "] LightSensorTD: Error parsing timeOfDay response: " + e.getMessage());
          promise.fail(e);
        }
      })
      .onFailure(err -> {
        System.err.println("[" + getId().getName() + "] LightSensorTD: Failed to retrieve simulation time: " + err.getMessage());
        promise.fail(err);
      });
    return promise.future();
  }

  private Future<Integer> getBlindsClosedPercentageAsync() {
    Promise<Integer> promise = Promise.promise();
    String blindsUrl = "http://localhost:8080/workspaces/lab308/artifacts/blinds308/status";
    WebClient client = WebClient.create(this.vertxInstance);

    client.postAbs(blindsUrl)
        .putHeader("Content-Type", "application/json")
        .putHeader("X-Agent-WebID", "http://localhost:8080/agents/alex")
        .send()
        .onSuccess(response -> {
          try {
            io.vertx.core.json.JsonObject statusJson = response.bodyAsJsonObject();
            if (statusJson != null && statusJson.containsKey("closedPercentage")) {
              Integer percentage = statusJson.getInteger("closedPercentage");
              promise.complete(percentage);
            } else {
              System.err.println("[" + getId().getName() + "] LightSensorTD: Invalid or missing closedPercentage in response from blinds: " + response.bodyAsString());
              promise.fail("Invalid response from blinds service for closedPercentage");
            }
          } catch (Exception e) {
            System.err.println("[" + getId().getName() + "] LightSensorTD: Error parsing blinds closedPercentage response: " + e.getMessage());
            promise.fail(e);
          }
        })
        .onFailure(err -> {
          System.err.println("[" + getId().getName() + "] LightSensorTD: Failed to retrieve blinds status: " + err.getMessage());
          promise.fail(err);
        });
    return promise.future();
  }

  private Future<Boolean> areLightsOffAsync() {
    Promise<Boolean> promise = Promise.promise();
    String lightsUrl = "http://localhost:8080/workspaces/lab308/artifacts/light308/status";
    WebClient client = WebClient.create(this.vertxInstance);

    client.postAbs(lightsUrl)
        .putHeader("Content-Type", "application/json")
        .putHeader("X-Agent-WebID", "http://localhost:8080/agents/alex")
        .send()
        .onSuccess(response -> {
          try {
            io.vertx.core.json.JsonObject statusJson = response.bodyAsJsonObject();
             if (statusJson != null && statusJson.containsKey("state")) {
              String state = statusJson.getString("state");
              promise.complete("off".equalsIgnoreCase(state));
            } else {
              System.err.println("[" + getId().getName() + "] LightSensorTD: Invalid or missing state in response from lights: " + response.bodyAsString());
              promise.fail("Invalid response from lights service for state");
            }
          } catch (Exception e) {
            System.err.println("[" + getId().getName() + "] LightSensorTD: Error parsing lights state response: " + e.getMessage());
            promise.fail(e);
          }
        })
        .onFailure(err -> {
          System.err.println("[" + getId().getName() + "] LightSensorTD: Failed to retrieve lights status: " + err.getMessage());
          promise.fail(err);
        });
    return promise.future();
  }

  private Future<Integer> getLightIntensityAsync() {
    Promise<Integer> promise = Promise.promise();
    String lightsUrl = "http://localhost:8080/workspaces/lab308/artifacts/light308/status";
    WebClient client = WebClient.create(this.vertxInstance);

    client.postAbs(lightsUrl)
        .putHeader("Content-Type", "application/json")
        .putHeader("X-Agent-WebID", "http://localhost:8080/agents/alex")
        .send()
        .onSuccess(response -> {
          try {
            io.vertx.core.json.JsonObject statusJson = response.bodyAsJsonObject();
            if (statusJson != null && statusJson.containsKey("lightIntensity")) {
              Integer intensity = statusJson.getInteger("lightIntensity");
              promise.complete(intensity);
            } else {
              System.err.println("[" + getId().getName() + "] LightSensorTD: Invalid or missing lightIntensity in response from lights: " + response.bodyAsString());
              promise.fail("Invalid response from lights service for lightIntensity");
            }
          } catch (Exception e) {
            System.err.println("[" + getId().getName() + "] LightSensorTD: Error parsing lights lightIntensity response: " + e.getMessage());
            promise.fail(e);
          }
        })
        .onFailure(err -> {
          System.err.println("[" + getId().getName() + "] LightSensorTD: Failed to retrieve lights lightIntensity: " + err.getMessage());
          promise.fail(err);
        });
    return promise.future();
  }
  
  @Override
  public void startPeriodicTasks() {
    if (this.vertxInstance != null && this.httpConfig != null && this.dispatcherMessagebox != null) {
        vertxInstance.setPeriodic(LUMINOSITY_CHECK_INTERVAL, timerId -> {
            Future<Integer> blindsClosedPercentageFuture = getBlindsClosedPercentageAsync();
            Future<Boolean> lightsOffFuture = areLightsOffAsync();
            Future<Integer> lightIntensityFuture = getLightIntensityAsync();
            Future<Long> timeFuture = retrieveCurrentSimulationTimeAsync();

            Future.all(blindsClosedPercentageFuture, lightsOffFuture, lightIntensityFuture, timeFuture)
                .onSuccess(compositeResult -> {
                    Integer blindsClosedPercentage = compositeResult.resultAt(0);
                    boolean lightsOff = compositeResult.resultAt(1);
                    Integer lightIntensityPercent = compositeResult.resultAt(2);
                    long currentTime = compositeResult.resultAt(3);

                    double newLuminosityValue = 0.0;

                    if (blindsClosedPercentage != null) {
                        double sunlightIntensity = computeSunlightIntensity(currentTime / 60.0);
                        double openPercentageFactor = (100.0 - blindsClosedPercentage) / 100.0;
                        newLuminosityValue += sunlightIntensity * openPercentageFactor;
                    } else {
                        System.err.println("[" + getId().getName() + "] LightSensorTD: Blinds closed percentage was null, assuming no sunlight.");
                    }

                    if (!lightsOff) {
                        if (lightIntensityPercent != null) {
                            newLuminosityValue += (lightIntensityPercent / 100.0) * MAX_LUMINOSITY_FROM_LIGHTS;
                        } else {
                             System.err.println("[" + getId().getName() + "] LightSensorTD: Light intensity was null, not adding light contribution.");
                        }
                    }

                    double currentLuminosityValue = (Double) this.getObsProperty("luminosity").getValue();

                    if (Math.abs(newLuminosityValue - currentLuminosityValue) > 0.001) {
                        this.getObsProperty("luminosity").updateValue(newLuminosityValue);

                        String currentWorkspaceName = getId().getWorkspaceId().getName();
                        String currentArtifactName = getId().getName();
                        String artifactUri = this.httpConfig.getArtifactUri(currentWorkspaceName, currentArtifactName);
                        String propertyName = "luminosity";
                        String propertyUri = artifactUri + "/props/" + propertyName;
                        String triggerUri = propertyUri;

                        int newLuminosityIntForNotification = (int) Math.round(newLuminosityValue);

                        JsonObject payload = Json.createObjectBuilder()
                            .add("artifactUri", artifactUri)
                            .add("propertyUri", propertyUri)
                            .add("value", newLuminosityIntForNotification)
                            .add("valueTypeUri", "http://www.w3.org/2001/XMLSchema#integer")
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
                })
                .onFailure(err -> {
                    System.err.println("[" + getId().getName() + "] LightSensorTD: Error during periodic luminosity check (one or more async calls failed): " + err.getMessage());
                });
        });
    } else {
        String reason = "";
        if (this.vertxInstance == null) reason += "Vert.x instance not initialized. ";
        if (this.httpConfig == null) reason += "HttpInterfaceConfig not initialized. ";
        if (this.dispatcherMessagebox == null) reason += "DispatcherMessagebox not initialized. ";
        System.err.println("[" + getId().getName() + "] LightSensorTD: Cannot start periodic tasks. " + reason.trim());
    }
  }

  /**
   * Retrieves the current luminosity measurement.
   * This value is updated periodically by an internal task.
   */
  @OPERATION
  public void getLuminosity(final OpFeedbackParam<String> jsonLuminosity) {
    double currentLuminosityDouble = (Double) this.getObsProperty("luminosity").getValue();
    int currentLuminosityInt = (int) Math.round(currentLuminosityDouble);
    io.vertx.core.json.JsonObject result = new io.vertx.core.json.JsonObject().put("luminosity", currentLuminosityInt);
    jsonLuminosity.set(result.toString());
  }
  
  @Override
  protected void registerInteractionAffordances() {
    this.setSecurityScheme(SecurityScheme.getNoSecurityScheme());
    
    this.registerActionAffordance(
      "http://example.org/MeasurementCommand",
      "getLuminosity",
      "luminosity",
      null,
      new ObjectSchema.Builder()
          .addProperty("luminosity", new IntegerSchema.Builder().build())
          .addRequiredProperties("luminosity")
          .build()
    );
  }
}

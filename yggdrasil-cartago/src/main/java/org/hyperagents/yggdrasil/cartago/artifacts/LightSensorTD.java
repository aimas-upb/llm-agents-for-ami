package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.schemas.NumberSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;

import java.util.concurrent.CompletableFuture;

import javax.json.Json;
import javax.json.JsonObject;
import io.vertx.core.Vertx;
import io.vertx.ext.web.client.WebClient;
import io.vertx.ext.web.client.HttpResponse;
import io.vertx.core.buffer.Buffer;

import java.util.concurrent.ExecutionException;

/**
 * LightSensor TD Artifact, measures luminosity.
 * Serves as an exemplary Hypermedia Artifact using the TD ontology.
 */
public class LightSensorTD extends HypermediaTDArtifact implements VertxInjectable {
  private Vertx vertxInstance;

  @Override
  public void setVertx(Vertx vertx) {
    this.vertxInstance = vertx;
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

  private long retrieveCurrentSimulationTime() {
    String clockUrl = "http://localhost:8080/workspaces/lab308/artifacts/clock308/timeOfDay";
    WebClient client = WebClient.create(this.vertxInstance);
    io.vertx.core.json.JsonObject requestBody = new io.vertx.core.json.JsonObject();
  
    CompletableFuture<HttpResponse<Buffer>> future = new CompletableFuture<>();
    client.postAbs(clockUrl)
      .putHeader("Content-Type", "application/json")
      .putHeader("X-Agent-WebID", "http://localhost:8080/agents/alex")
      .sendJsonObject(requestBody, ar -> {
        if (ar.succeeded()) {
          future.complete(ar.result());
        } else {
          future.completeExceptionally(ar.cause());
        }
      });

    HttpResponse<Buffer> response = future.join();
    io.vertx.core.json.JsonObject timeJson = response.bodyAsJsonObject();
    return timeJson.getLong("timeOfDay");
  }

  private boolean areBlindsClosed() {
    String blindsUrl = "http://localhost:8080/workspaces/lab308/artifacts/blinds308/status";
    WebClient client = WebClient.create(this.vertxInstance);
    CompletableFuture<HttpResponse<Buffer>> future = new CompletableFuture<>();
    
    client.postAbs(blindsUrl)
        .putHeader("Content-Type", "application/json")
        .putHeader("X-Agent-WebID", "http://localhost:8080/agents/alex")
        .send(ar -> {
            if (ar.succeeded()) {
                future.complete(ar.result());
            } else {
                future.completeExceptionally(ar.cause());
            }
        });
    
    HttpResponse<Buffer> response = future.join();
    io.vertx.core.json.JsonObject statusJson = response.bodyAsJsonObject();
    String state = statusJson.getString("state");
    System.out.println("Blinds state retrieved: " + state);
    
    return "closed".equalsIgnoreCase(state);
  }

  private boolean areLightsClosed() {
    String blindsUrl = "http://localhost:8080/workspaces/lab308/artifacts/light308/status";
    WebClient client = WebClient.create(this.vertxInstance);
    CompletableFuture<HttpResponse<Buffer>> future = new CompletableFuture<>();
    
    client.postAbs(blindsUrl)
        .putHeader("Content-Type", "application/json")
        .putHeader("X-Agent-WebID", "http://localhost:8080/agents/alex")
        .send(ar -> {
            if (ar.succeeded()) {
                future.complete(ar.result());
            } else {
                future.completeExceptionally(ar.cause());
            }
        });
    
    HttpResponse<Buffer> response = future.join();
    io.vertx.core.json.JsonObject statusJson = response.bodyAsJsonObject();
    String state = statusJson.getString("state");
    System.out.println("Lights state retrieved: " + state);
    
    return "off".equalsIgnoreCase(state);
  }
  
  /**
   * Retrieves the current luminosity measurement.
   */
  @OPERATION
  public void getLuminosity(final OpFeedbackParam<String> jsonLuminosity) {
    if (this.vertxInstance != null) {
      double lumValue = 0.0;
      
      if (!areBlindsClosed()) {
        long currentTime = retrieveCurrentSimulationTime();
        lumValue += computeSunlightIntensity(currentTime / 60.0);
      }

      if (!areLightsClosed()) {
        lumValue += 500.0;
      }
      this.getObsProperty("luminosity").updateValue(lumValue);
    }

  double lumValue = this.getObsProperty("luminosity").doubleValue();
  io.vertx.core.json.JsonObject result = new io.vertx.core.json.JsonObject().put("luminosity", lumValue);
  jsonLuminosity.set(result.toString());
  System.out.println("Luminosity is " + result);
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
          .addProperty("luminosity", new NumberSchema.Builder().build())
          .addRequiredProperties("luminosity")
          .build()
    );
  }
}

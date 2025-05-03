package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.schemas.BooleanSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;

import java.util.concurrent.CompletableFuture;

import javax.json.Json;
import javax.json.JsonObject;
import io.vertx.core.Vertx;
import io.vertx.core.buffer.Buffer;
import io.vertx.ext.web.client.HttpResponse;
import io.vertx.ext.web.client.WebClient;

/**
 * PeoplePresenceSensor TD Artifact, detects human presence.
 * Serves as an exemplary Hypermedia Artifact using the TD ontology.
 */
public class PeoplePresenceSensorTD extends HypermediaTDArtifact implements VertxInjectable {
  private Vertx vertxInstance;

  public void init() {
    this.defineObsProperty("presence", false);
  }
  
  public void init(final boolean presence) {
    this.defineObsProperty("presence", presence);
  }

  @Override
  public void setVertx(Vertx vertx) {
    this.vertxInstance = vertx;
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
  
  /**
   * Retrieves the current presence detection status.
   */
  @OPERATION
  public void getPresence(final OpFeedbackParam<String> jsonPresence) {
    long currentTime = retrieveCurrentSimulationTime();
    double hour = currentTime / 60.0;

    boolean presence = false;

    if (hour >= 8 && hour <= 18) {
      presence = Math.random() < 0.9;
    }

    this.updateObsProperty("presence", presence);
    JsonObject json = Json.createObjectBuilder()
        .add("presence", presence)
        .build();
    jsonPresence.set(json.toString());
    System.out.println("Presence detected: " + json);
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

package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.schemas.StringSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;
import javax.json.Json;
import javax.json.JsonObject;
import java.util.Arrays;
import java.util.HashSet;

/**
 * Blinds TD Artifact, has an open/closed state that can be toggled.
 * Serves as an exemplary Hypermedia Artifact, using TD as its ontology.
 */
public class BlindsTD extends HypermediaTDArtifact {

  public void init() {
    this.defineObsProperty("state", "closed");
  }
  
  public void init(final String state) {
    this.defineObsProperty("state", state);
  }
  
  /**
   * Retrieves the internal state of the blinds.
   */
  @OPERATION
  public void getStatus(final OpFeedbackParam<String> jsonStatus) {
    String stateValue = this.getObsProperty("state").stringValue();
    JsonObject json = Json.createObjectBuilder()
        .add("state", stateValue)
        .build();
    jsonStatus.set(json.toString());
    System.out.println("Status is " + json);
  }
  
  /**
   * Toggles the internal state of the blinds.
   */
  @OPERATION
  public void toggle() {
    final var prop = this.getObsProperty("state");
    if (prop.stringValue().equals("open")) {
      prop.updateValue("closed");
    } else {
      prop.updateValue("open");
    }
    System.out.println("State toggled, current state is " + prop.stringValue());
  }
  
  @Override
  protected void registerInteractionAffordances() {
    this.setSecurityScheme(SecurityScheme.getNoSecurityScheme());
    
    this.registerActionAffordance("http://example.org/StatusCommand", "getStatus", "status", 
        null,
        new ObjectSchema.Builder()
          .addProperty("state", new StringSchema.Builder()
            .addEnum(new HashSet<>(Arrays.asList("open", "closed")))
            .build())
          .addRequiredProperties("state")
          .build()
    );
    
    this.registerActionAffordance("http://example.org/ToggleCommand", "toggle", "toggle");
  }
}

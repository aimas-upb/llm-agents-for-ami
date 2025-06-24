package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.IntegerSchema;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.schemas.StringSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;
import javax.json.Json;
import javax.json.JsonObject;
import java.util.Arrays;
import java.util.HashSet;

/**
 * Blinds TD Artifact, has a closedPercentage that can be set.
 * Serves as an exemplary Hypermedia Artifact, using TD as its ontology.
 */
public class BlindsTD extends HypermediaTDArtifact {

  public void init() {
    this.defineObsProperty("closedPercentage", 100);
  }

  public void init(final Integer percentage) {
    this.defineObsProperty("closedPercentage", percentage);
  }
  
  /**
   * Retrieves the current closed percentage of the blinds.
   */
  @OPERATION
  public void getStatus(final OpFeedbackParam<String> jsonStatus) {
    Integer closedPercentageValue = this.getObsProperty("closedPercentage").intValue();
    JsonObject json = Json.createObjectBuilder()
        .add("closedPercentage", closedPercentageValue)
        .build();
    jsonStatus.set(json.toString());
    System.out.println("Status is " + json);
  }
  
  /**
   * Sets the closed percentage of the blinds.
   */
  @OPERATION
  public void setClosedPercentage(final Integer percentage) {
    final var prop = this.getObsProperty("closedPercentage");
    prop.updateValue(percentage);
    System.out.println("closedPercentage set to " + percentage);
  }
  
  @Override
  protected void registerInteractionAffordances() {
    this.setSecurityScheme(SecurityScheme.getNoSecurityScheme());
    
    this.registerActionAffordance("http://example.org/StatusCommand", "getStatus", "status",
        null,
        new ObjectSchema.Builder()
          .addProperty("closedPercentage", new IntegerSchema.Builder()
            .addEnum(new HashSet<>(Arrays.asList("0", "25", "50", "100")))
            .build())
          .addRequiredProperties("closedPercentage")
          .build()
    );
    
    this.registerActionAffordance(
        "http://example.org/SetClosedPercentageCommand", 
        "setClosedPercentage", 
        "setClosedPercentage",
        new ObjectSchema.Builder()
            .addProperty("closedPercentage", new IntegerSchema.Builder()
                .addEnum(new HashSet<>(Arrays.asList("0", "25", "50", "100")))
                .build())
            .addRequiredProperties("closedPercentage")
            .build(),
        null
    );
  }
}

package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.ArraySchema;
import ch.unisg.ics.interactions.wot.td.schemas.IntegerSchema;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.schemas.StringSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;
import ch.unisg.ics.interactions.wot.td.schemas.DataSchema.JsonSchemaBuilder;
import ch.unisg.ics.interactions.wot.td.vocabularies.JSONSchema;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.List;

import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;

import java.io.StringReader;

/**
 * HueLamp TD Artifact, has an on/off state that can be toggled and a color 
 * that can be set. Serves as an exemplary Hypermedia Artifact, uses TD as its ontology
 */
public class HueLampTD extends HypermediaTDArtifact {
    
  public void init() {
    this.defineObsProperty("state", "off");
    this.defineObsProperty("color", "blue");
  }

  public void init(final String state, final String color) {
    this.defineObsProperty("state", state);
    this.defineObsProperty("color", color);
  }

  /**
   * Retrieves the internal state of the lamp.
   */
  @OPERATION
  public void getStatus(final OpFeedbackParam<String> jsonStatus) {
    String stateValue = this.getObsProperty("state").stringValue();
    String colorValue = this.getObsProperty("color").stringValue();

    JsonObject json = Json.createObjectBuilder()
        .add("state", stateValue)
        .add("color", colorValue)
        .build();
    jsonStatus.set(json.toString());
    System.out.println("Status is " + json);
}

  /**
   * Toggles the internal state of the lamp.
   */
  @OPERATION
  public void toggle() {
    final var prop = this.getObsProperty("state");
    if (prop.stringValue().equals("on")) {
      prop.updateValue("off");
    } else {
      prop.updateValue("on");
    }
    System.out.println("state toggled, current state is " + prop.stringValue());
  }

  /**
   * Sets the internal color of the lamp.
   */
  @OPERATION
  public void setColor(final String color) {
    final var prop = this.getObsProperty("color");
    prop.updateValue(color);
    System.out.println("color set to " + color);
  }

  @Override
  protected void registerInteractionAffordances() {
    this.setSecurityScheme(SecurityScheme.getNoSecurityScheme());

    this.registerActionAffordance("http://example.org/StatusCommand", "getStatus", "status",
    null,
    new ObjectSchema.Builder()
      .addProperty("state", new StringSchema.Builder()
        .addEnum(new HashSet<>(Arrays.asList("on", "off"))).build())
      .addProperty("color", new StringSchema.Builder()
        .addEnum(new HashSet<>(Arrays.asList("red", "green", "blue"))).build())
      .addRequiredProperties("state", "color")
      .build()
    );

    this.registerActionAffordance(
      "http://example.org/ToggleCommand",
      "toggle",
      "toggle"
    );

    this.registerActionAffordance(
      "http://example.org/ColorCommand",
      "setColor",
      "color",
      new ObjectSchema.Builder()
        .addProperty("color", new StringSchema.Builder()
          .addEnum(new HashSet<>(Arrays.asList("red", "green", "blue"))).build())
        .addRequiredProperties("color")
        .build(),
      null
    );
  }
}

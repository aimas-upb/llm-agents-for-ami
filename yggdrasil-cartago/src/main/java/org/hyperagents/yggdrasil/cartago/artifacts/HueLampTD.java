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
 * HueLamp TD Artifact, has an on/off state, color, and lightIntensity.
 * Serves as an exemplary Hypermedia Artifact, uses TD as its ontology
 */
public class HueLampTD extends HypermediaTDArtifact {
    
  public void init() {
    this.defineObsProperty("state", "off");
    this.defineObsProperty("color", "blue");
    this.defineObsProperty("lightIntensity", 25);
  }

  public void init(final String state, final String color, final Integer lightIntensity) {
    this.defineObsProperty("state", state);
    this.defineObsProperty("color", color);
    this.defineObsProperty("lightIntensity", lightIntensity);
  }

  /**
   * Retrieves the internal state of the lamp.
   */
  @OPERATION
  public void getStatus(final OpFeedbackParam<String> jsonStatus) {
    String stateValue = this.getObsProperty("state").stringValue();
    String colorValue = this.getObsProperty("color").stringValue();
    Integer lightIntensityValue = this.getObsProperty("lightIntensity").intValue();

    JsonObject json = Json.createObjectBuilder()
        .add("state", stateValue)
        .add("color", colorValue)
        .add("lightIntensity", lightIntensityValue)
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

  /**
   * Sets the internal light intensity of the lamp.
   */
  @OPERATION
  public void setLightIntensity(final Integer lightIntensity) {
    final var prop = this.getObsProperty("lightIntensity");
    prop.updateValue(lightIntensity);
    System.out.println("lightIntensity set to " + lightIntensity);
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
      .addProperty("lightIntensity", new IntegerSchema.Builder()
        .addEnum(new HashSet<>(Arrays.asList("25", "50", "75", "100"))).build())
      .addRequiredProperties("state", "color", "lightIntensity")
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

    this.registerActionAffordance(
      "http://example.org/LightIntensityCommand",
      "setLightIntensity",
      "lightIntensity",
      new ObjectSchema.Builder()
        .addProperty("lightIntensity", new IntegerSchema.Builder()
          .addEnum(new HashSet<>(Arrays.asList("25", "50", "75", "100"))).build())
        .addRequiredProperties("lightIntensity")
        .build(),
      null
    );
  }
}

package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.IntegerSchema;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;
import io.vertx.core.Vertx;
import javax.json.Json;
import javax.json.JsonObject;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

/**
 * ClockTD is a simulation clock artifact that simulates the time of day in a virtual office.
 * It maintains an observable property "timeOfDay" (in minutes since midnight) and updates it periodically.
 */
public class ClockTD extends HypermediaTDArtifact implements VertxInjectable, NeedsPeriodic {

    private Vertx vertxInstance;
    private long timeOfDay = 480;
    private final long tickIncrement = 1;
    private final long tickInterval = 1000;

    @Override
    public void setVertx(Vertx vertx) {
        this.vertxInstance = vertx;
    }

    public void init() {
        this.defineObsProperty("timeOfDay", timeOfDay);
    }

    public void init(final long timeOfDay) {
        this.timeOfDay = timeOfDay;
        this.defineObsProperty("timeOfDay", timeOfDay);
    }

    @Override
    public void startPeriodicTasks() {
        if (this.vertxInstance != null) {
            vertxInstance.setPeriodic(tickInterval, timerId -> {
                timeOfDay = (timeOfDay + tickIncrement) % 1440;
                this.getObsProperty("timeOfDay").updateValue(timeOfDay);
                //System.out.println("Simulation time updated: " + formatTime(timeOfDay));
            });
        } else {
            System.err.println("Vert.x instance not injected in ClockTD!");
        }
    }

    /**
     * Retrieves the current simulation time as a JSON object.
     * Returns both the numeric time (minutes since midnight) and a formatted string (HH:mm).
     */
    @OPERATION
    public void getTime(final OpFeedbackParam<String> jsonTime) {
        long currentTime = this.getObsProperty("timeOfDay").longValue();
        String formattedTime = formatTime(currentTime);
        JsonObject json = Json.createObjectBuilder()
                .add("timeOfDay", currentTime)
                .add("formattedTime", formattedTime)
                .build();
        jsonTime.set(json.toString());
        //System.out.println("Current simulation time: " + json);
    }

    /**
     * Helper method to convert minutes since midnight into a formatted time string.
     */
    private String formatTime(long minutesSinceMidnight) {
        int hours = (int) (minutesSinceMidnight / 60);
        int minutes = (int) (minutesSinceMidnight % 60);
        LocalTime time = LocalTime.of(hours, minutes);
        return time.format(DateTimeFormatter.ofPattern("HH:mm"));
    }

    @Override
    protected void registerInteractionAffordances() {
        this.setSecurityScheme(SecurityScheme.getNoSecurityScheme());
        this.registerActionAffordance(
            "http://example.org/GetClockTime",
            "getTime",
            "timeOfDay",
            null,
            new ObjectSchema.Builder()
                .addProperty("timeOfDay", new IntegerSchema.Builder().build())
                .addRequiredProperties("timeOfDay")
                .build()
        );
    }
}

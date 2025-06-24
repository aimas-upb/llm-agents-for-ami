package org.hyperagents.yggdrasil.cartago.artifacts;

import cartago.OPERATION;
import cartago.OpFeedbackParam;
import ch.unisg.ics.interactions.wot.td.schemas.IntegerSchema;
import ch.unisg.ics.interactions.wot.td.schemas.ObjectSchema;
import ch.unisg.ics.interactions.wot.td.security.SecurityScheme;
import io.vertx.core.Vertx;
import org.hyperagents.yggdrasil.eventbus.messageboxes.HttpNotificationDispatcherMessagebox;
import org.hyperagents.yggdrasil.eventbus.messages.HttpNotificationDispatcherMessage;
import org.hyperagents.yggdrasil.utils.HttpInterfaceConfig;
import org.hyperagents.yggdrasil.utils.WebSubConfig;
import javax.json.Json;
import javax.json.JsonObject;
import java.time.Instant;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

/**
 * ClockTD is a simulation clock artifact that simulates the time of day in a virtual office.
 * It maintains an observable property "timeOfDay" (in minutes since midnight) and updates it periodically.
 */
public class ClockTD extends HypermediaTDArtifact implements VertxInjectable, NeedsPeriodic {

    private Vertx vertxInstance;
    private HttpInterfaceConfig httpConfig;
    private WebSubConfig webSubConfig;
    private HttpNotificationDispatcherMessagebox dispatcherMessagebox;
    private static final String DEFAULT_CONFIG_VALUE = "default";

    private long timeOfDay = 480;
    private final long tickIncrement = 1;
    private final long tickInterval = 1000;

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
            if (this.webSubConfig != null) {
                 this.dispatcherMessagebox = new HttpNotificationDispatcherMessagebox(
                    this.vertxInstance.eventBus(),
                    this.webSubConfig
                );
            } else {
                System.err.println("ClockTD: WebSubConfig not found in shared data.");
            }
            if (this.httpConfig == null) {
                 System.err.println("ClockTD: HttpInterfaceConfig not found in shared data.");
            }
        }
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
        if (this.vertxInstance != null && this.httpConfig != null && this.dispatcherMessagebox != null) {
            vertxInstance.setPeriodic(tickInterval, timerId -> {
                timeOfDay = (timeOfDay + tickIncrement) % 1440;
                this.getObsProperty("timeOfDay").updateValue(timeOfDay);

                String currentWorkspaceName = getId().getWorkspaceId().getName();
                String currentArtifactName = getId().getName();

                String artifactUri = this.httpConfig.getArtifactUri(currentWorkspaceName, currentArtifactName);
                String propertyName = "timeOfDay";
                String propertyUri = artifactUri + "/props/" + propertyName;
                String triggerUri = propertyUri;
                
                String formattedTimeValue = formatTime(this.timeOfDay);
                String valueXsdType = "http://www.w3.org/2001/XMLSchema#string";

                JsonObject payload = Json.createObjectBuilder()
                  .add("artifactUri", artifactUri)
                  .add("propertyUri", propertyUri)
                  .add("value", formattedTimeValue)
                  .add("valueTypeUri", valueXsdType)
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
            });
        } else {
            System.err.println("ClockTD: Vert.x instance, httpConfig, or dispatcherMessagebox not initialized properly. Cannot start periodic tasks for manual notification.");
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

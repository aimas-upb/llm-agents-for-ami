package org.hyperagents.yggdrasil.cartago.entities;

import cartago.ArtifactObsProperty; // Use specific type
import cartago.CartagoEvent;
import cartago.ICartagoCallback;
import cartago.ObsProperty;
import cartago.events.ArtifactObsEvent;
import cartago.util.agent.Percept;
import io.vertx.core.json.JsonObject;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import org.hyperagents.yggdrasil.cartago.HypermediaArtifactRegistry;
import org.hyperagents.yggdrasil.eventbus.messageboxes.HttpNotificationDispatcherMessagebox;
import org.hyperagents.yggdrasil.eventbus.messages.HttpNotificationDispatcherMessage;
import org.hyperagents.yggdrasil.utils.HttpInterfaceConfig;

/**
 * Implementation for the CartagoCallback. Is used to define the actions taken when a CartagoEvent
 * is triggered.
 */
public class NotificationCallback implements ICartagoCallback {
  private final HttpInterfaceConfig httpConfig;
  private final HttpNotificationDispatcherMessagebox messagebox;
  private final String workspaceName;
  private final String artifactName;
  private final Map<String, String> lastActionPerArtifact;
  private final HypermediaArtifactRegistry artifactRegistry;

  private static final Map<String, Long> lastNotificationTimestamps = new HashMap<>();
  private static final long DEBOUNCE_DELAY_MS = 100;

  /**
   * Constructor for the Notification Callback.
   *
   * @param httpConfig    the httpConfig of the Yggdrasil instance.
   * @param messagebox    the HttpNotification messagebox.
   * @param workspaceName the workspaceName.
   * @param artifactName  the artifactName.
   * @param lastActionPerArtifact map tracking last action per artifact.
   * @param artifactRegistry registry for artifact metadata.
   */
  public NotificationCallback(
      final HttpInterfaceConfig httpConfig,
      final HttpNotificationDispatcherMessagebox messagebox,
      final String workspaceName,
      final String artifactName,
      final Map<String, String> lastActionPerArtifact,
      final HypermediaArtifactRegistry artifactRegistry
  ) {
    this.httpConfig = httpConfig;
    this.messagebox = messagebox;
    this.workspaceName = workspaceName;
    this.artifactName = artifactName;
    this.lastActionPerArtifact = lastActionPerArtifact;
    this.artifactRegistry = artifactRegistry;
  }

  @Override
  public void notifyCartagoEvent(final CartagoEvent event) {
    if (event instanceof ArtifactObsEvent e) {
      final var percept = new Percept(e);

      if (percept.hasSignal()) {
        Stream.of(Optional.ofNullable(percept.getSignal()))
            .flatMap(Optional::stream)
            .forEach(p -> System.err.println(
                "Warning: Received signal in NotificationCallback, not sending JSON: " + p
            ));
        return;
      }

      assert percept.getArtifactSource().getName().equals(artifactName)
          : "Artifact name mismatch";
      assert percept.getArtifactSource().getWorkspaceId().getName().equals(workspaceName)
          : "Workspace name mismatch";

      Stream.of(
              Optional.ofNullable(percept.getPropChanged()),
              Optional.ofNullable(percept.getAddedProperties()),
              Optional.ofNullable(percept.getRemovedProperties())
            )
            .flatMap(Optional::stream)
            .flatMap(Arrays::stream)
            .forEach((ArtifactObsProperty p) -> {
              final String propertyName = p.getName();
              final String artifactUri =
                  this.httpConfig.getArtifactUri(this.workspaceName, this.artifactName);
              final String debounceKey = artifactUri + "/" + propertyName;
              final long currentTime = System.currentTimeMillis();

              synchronized (lastNotificationTimestamps) {
                final Long lastTimestamp = lastNotificationTimestamps.get(debounceKey);

                if (lastTimestamp != null && (currentTime - lastTimestamp <= DEBOUNCE_DELAY_MS)) {
                  return;
                }
                lastNotificationTimestamps.put(debounceKey, currentTime);
              }

              final String artifactMapKey = this.workspaceName + "/" + this.artifactName;
              final String actionName =
                  this.lastActionPerArtifact.getOrDefault(artifactMapKey, "unknown");

              final String triggerUri = "unknown".equals(actionName)
                  ? "urn:unknown"
                  : artifactUri + "/actions/" + actionName;

              JsonObject payload = buildJsonPropertyPayload(artifactUri, p, triggerUri);

              this.messagebox.sendMessage(
                new HttpNotificationDispatcherMessage.ArtifactObsPropertyUpdated(
                    artifactUri,
                    payload.encode()
                )
               );
          });
    }
  }

  private JsonObject buildJsonPropertyPayload(String artifactUri, ArtifactObsProperty property, String triggerUri) {
    String propertyName = property.getName();
    Object value = property.getValue();
    String xsdType = getXsdType(value);
    String propertyUri = artifactUri + "/props/" + propertyName;

    return new JsonObject()
      .put("artifactUri", artifactUri)
      .put("propertyUri", propertyUri)
      .put("value", value)
      .put("valueTypeUri", xsdType)
      .put("timestamp", DateTimeFormatter.ISO_INSTANT.format(Instant.now()))
      .put("triggerUri", triggerUri);
  }

  private String getXsdType(Object value) {
    if (value instanceof Integer || value instanceof Long) {
      return "http://www.w3.org/2001/XMLSchema#integer";
    } else if (value instanceof Double || value instanceof Float) {
      return "http://www.w3.org/2001/XMLSchema#double";
    } else if (value instanceof Boolean) {
      return "http://www.w3.org/2001/XMLSchema#boolean";
    } else {
      return "http://www.w3.org/2001/XMLSchema#string";
    }
  }
}

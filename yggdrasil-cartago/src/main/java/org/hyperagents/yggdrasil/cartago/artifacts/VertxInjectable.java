package org.hyperagents.yggdrasil.cartago.artifacts;
import io.vertx.core.Vertx;

public interface VertxInjectable {
    void setVertx(Vertx vertx);
}

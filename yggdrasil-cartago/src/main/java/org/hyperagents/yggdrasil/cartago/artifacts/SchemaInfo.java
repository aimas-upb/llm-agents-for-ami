package org.hyperagents.yggdrasil.cartago.artifacts;

public class SchemaInfo {
    private final String dataType;
    private final int paramCount;
  
    public SchemaInfo(String dataType, int paramCount) {
      this.dataType = dataType;
      this.paramCount = paramCount;
    }
  
    public String getDataType() {
      return dataType;
    }
  
    public int getParamCount() {
      return paramCount;
    }
  }
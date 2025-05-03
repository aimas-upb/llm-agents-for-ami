import React, { useState, useEffect, useCallback } from 'react';
import * as rdflib from 'rdflib';

const JS = rdflib.Namespace('https://www.w3.org/2019/wot/json-schema#');
const RDF = rdflib.Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#');

function ArtifactActionPayload({ actionName, inputSchemaNode, store, onPayloadChange }) {
  const [payload, setPayload] = useState('');
  const [payloadType, setPayloadType] = useState('json');
  const [schemaProperties, setSchemaProperties] = useState([]);

  // Extract properties from a given schema node
  const extractProperties = useCallback((schemaNode) => {
    const properties = [];
    const propertyNodes = store.match(schemaNode, JS('properties'), undefined);
    
    propertyNodes.forEach(pn => {
      const propertyName = store.any(pn.object, JS('propertyName'))?.value;
      const enumValues = store.match(pn.object, JS('enum'), undefined).map(e => e.object.value);
      const isRequired = store.match(schemaNode, JS('required'), rdflib.lit(propertyName)).length > 0;
      const propType = store.any(pn.object, RDF('type'))?.value;

      if (propertyName) {
        properties.push({
          name: propertyName,
          type: propType?.split('#').pop() || 'string',
          enum: enumValues,
          required: isRequired
        });
      }
    });

    return properties;
  }, [store]);

  useEffect(() => {
    if (!inputSchemaNode || !store) return;

    try {
      let properties = [];
      
      // Check if the schema is an ObjectSchema
      const isObjectSchema = store.match(
        inputSchemaNode, 
        RDF('type'), 
        JS('ObjectSchema')
      ).length > 0;

      if (isObjectSchema) {
        properties = extractProperties(inputSchemaNode);
      } else {
        // Handle other schema types (e.g. ArraySchema)
        const items = store.match(inputSchemaNode, JS('items'), undefined);
        items.forEach(item => {
          properties = [...properties, ...extractProperties(item.object)];
        });
      }

      setSchemaProperties(properties);
    } catch (error) {
      console.error('Error parsing schema:', error);
    }
  }, [inputSchemaNode, store, extractProperties]);

  // Generate a JSON example based on the schema properties
  const generateJSONExample = useCallback(() => {
    if (schemaProperties.length === 0) return "{}";
    const exampleObject = {};
    schemaProperties.forEach(prop => {
      if (prop.enum?.length > 0) {
        exampleObject[prop.name] = prop.enum[0];
      } else if (prop.type === 'number') {
        exampleObject[prop.name] = 0;
      } else if (prop.type === 'boolean') {
        exampleObject[prop.name] = false;
      } else {
        exampleObject[prop.name] = "";
      }
    });
    return JSON.stringify(exampleObject, null, 2);
  }, [schemaProperties]);

  // Generate an RDF example based on the schema properties
  const generateRDFExample = useCallback(() => {
    if (schemaProperties.length === 0) return "";
    let example = "@prefix ex: <http://example.com/> .\n";
    example += "ex:subject ";
    schemaProperties.forEach((prop, idx) => {
      let value;
      if (prop.enum && prop.enum.length > 0) {
        value = `"${prop.enum[0]}"`;
      } else if (prop.type === 'number') {
        value = "0";
      } else if (prop.type === 'boolean') {
        value = "false";
      } else {
        value = `"${prop.name}_value"`;
      }
      if (idx < schemaProperties.length - 1) {
        example += `ex:${prop.name} ${value} ;\n`;
      } else {
        example += `ex:${prop.name} ${value} .`;
      }
    });
    return example;
  }, [schemaProperties]);

  // Update the payload example when the schema or payload type changes
  useEffect(() => {
    if (schemaProperties.length === 0) return;
  
    const example = payloadType === 'json' 
      ? generateJSONExample() 
      : generateRDFExample();
    
    setPayload(example);
    onPayloadChange(actionName, { payload: example, payloadType });
  }, [schemaProperties, payloadType, generateJSONExample, generateRDFExample, actionName, onPayloadChange]);

  return (
    <div style={{ marginTop: '10px' }}>
      <label style={{ marginRight: '10px' }}>
        <input 
          type="radio" 
          name={`payloadType-${actionName}`} 
          value="json" 
          checked={payloadType === 'json'} 
          onChange={() => setPayloadType('json')}
        />
        JSON
      </label>
      <label style={{ marginRight: '10px' }}>
        <input 
          type="radio" 
          name={`payloadType-${actionName}`} 
          value="rdf" 
          checked={payloadType === 'rdf'} 
          onChange={() => setPayloadType('rdf')}
        />
        RDF
      </label>
      <br />
      <textarea
        value={payload}
        onChange={(e) => {
          const newPayload = e.target.value;
          setPayload(newPayload);
          onPayloadChange(actionName, { payload: newPayload, payloadType });
        }}
        placeholder={`Enter ${payloadType.toUpperCase()} payload`}
        rows={8}
        cols={60}
        style={{ 
          marginTop: '5px',
          fontFamily: payloadType === 'json' ? 'monospace' : 'inherit',
          whiteSpace: payloadType === 'rdf' ? 'pre' : 'normal'
        }}
      />
    </div>
  );
}

export default ArtifactActionPayload;

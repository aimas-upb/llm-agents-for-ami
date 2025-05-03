import React from 'react';
import ArtifactActionPayload from './ArtifactActionPayload';
import { getHumanReadableProperty, getHumanReadableValue, getPropertyType } from '../utils/schemaUtils';

function ArtifactDetails({ selectedArtifact, artifactDetails, handleExecuteAction, handlePayloadChange }) {
  return (
    <div style={{ marginTop: '20px' }}>
      <h2>Artifact Details: {selectedArtifact.name}</h2>

      <h3>Properties</h3>
      {artifactDetails.properties.length > 0 ? (
        <div className="properties-table">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Property</th>
                <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Value</th>
              </tr>
            </thead>
            <tbody>
              {artifactDetails.properties
                .filter(prop => {
                  return !prop.predicate.endsWith('type') && 
                         !prop.predicate.includes('hasActionAffordance') && 
                         !prop.predicate.includes('hasEventAffordance');
                })
                .map((prop, idx) => {
                  const propName = getHumanReadableProperty(prop.predicate);
                  const propType = getPropertyType(prop.predicate);
                  let propValue;
                  
                  if (propType === 'security') {
                    propValue = 'No Security';
                  } else if (propType === 'container') {
                    const match = prop.object.match(/\/workspaces\/([^/#]+)/);
                    propValue = match && match[1] ? match[1] : getHumanReadableValue(prop.object);
                  } else {
                    propValue = getHumanReadableValue(prop.object);
                  }
                  
                  return (
                    <tr key={idx} style={{ borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '8px', fontWeight: 'bold' }}>{propName}</td>
                      <td style={{ padding: '8px' }}>{propValue}</td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      ) : (
        <p>No properties available.</p>
      )}

      <h3>Actions</h3>
      {artifactDetails.actions.length > 0 ? (
        artifactDetails.actions.map((action, idx) => (
          <div key={idx} style={{ border: '1px solid #ccc', padding: '10px', marginBottom: '10px' }}>
            <p><strong>Name:</strong> {action.name}</p>
            <p><strong>Title:</strong> {action.title}</p>
            {action.forms?.length > 0 && (
              <>
                <p><strong>Forms:</strong></p>
                <ul>
                  {action.forms.map((form, fIdx) => (
                    <li key={fIdx}>
                      Method: {form.method}, Target: {form.target}, Content-Type: {form.contentType}
                    </li>
                  ))}
                </ul>
              </>
            )}
            {action.inputSchemaNode && (
              <ArtifactActionPayload 
                actionName={action.name}
                inputSchemaNode={action.inputSchemaNode}
                store={artifactDetails.store}
                onPayloadChange={handlePayloadChange}
              />
            )}
            <button onClick={() => handleExecuteAction(action)} style={{ marginTop: '10px' }}>
              Execute Action
            </button>
          </div>
        ))
      ) : (
        <p>No actions available.</p>
      )}

      <h3>Events</h3>
      {artifactDetails.events.length > 0 ? (
        <ul>
          {artifactDetails.events.map((event, idx) => (
            <li key={idx}>
              <strong>{event.name}</strong>: {event.title}
            </li>
          ))}
        </ul>
      ) : (
        <p>No events available.</p>
      )}
    </div>
  );
}

export default ArtifactDetails;

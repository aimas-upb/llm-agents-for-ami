import React from 'react';

function ArtifactList({ artifacts, handleArtifactClick }) {
  return (
    <>
      {artifacts.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h2>Artifacts</h2>
          <ul>
            {artifacts.map((artifact, index) => (
              <li
                key={index}
                style={{ cursor: 'pointer', padding: '4px 0', color: 'blue' }}
                onClick={() => handleArtifactClick(artifact)}
              >
                {artifact}
              </li>
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

export default ArtifactList;

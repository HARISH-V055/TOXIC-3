import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/index.css';

const root = document.getElementById('root');

if (!root) {
  throw new Error('Root element not found. Make sure index.html has <div id="root"></div>');
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

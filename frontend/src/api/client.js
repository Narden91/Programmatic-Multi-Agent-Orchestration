const API_BASE = '/api';

export async function sendQuery(query, apiKey, model) {
  const body = { query, model };
  if (apiKey) body.api_key = apiKey;

  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Request failed (${res.status})`);
  }

  return res.json();
}

/**
 * Single init call returning config + models in one request.
 * Falls back to separate calls if the endpoint is unavailable.
 */
export async function getInit() {
  const res = await fetch(`${API_BASE}/init`);
  if (!res.ok) throw new Error('Failed to fetch init');
  return res.json();
}

export async function getModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function getConfig() {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error('Failed to fetch config');
  return res.json();
}

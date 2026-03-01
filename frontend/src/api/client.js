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
 * Implements exponential backoff to handle slow Uvicorn startup
 */
export async function getInit(retries = 5, delay = 1000) {
  try {
    const res = await fetch(`${API_BASE}/init`);
    if (!res.ok) throw new Error('Failed to fetch init');
    return await res.json();
  } catch (error) {
    if (retries > 0) {
      console.log(`[MoE] Backend not ready yet, retrying in ${delay}ms... (${retries} attempts left)`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return getInit(retries - 1, delay);
    }
    throw error;
  }
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

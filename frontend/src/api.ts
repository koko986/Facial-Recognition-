import type { AnalyzeResponse, ExperimentResult, Person, RecognizeResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export function resolveMediaUrl(path: string) {
  if (path.startsWith("http")) return path;
  const clean = path.replace(/\\/g, "/");
  const dataIndex = clean.indexOf("data/");
  return dataIndex >= 0 ? `${API_URL}/${clean.slice(dataIndex)}` : `${API_URL}/${clean}`;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Request failed." }));
    throw new Error(body.detail ?? "Request failed.");
  }
  return response.json() as Promise<T>;
}

export async function fetchPeople() {
  return parseResponse<Person[]>(await fetch(`${API_URL}/api/people`));
}

export async function fetchExperiments() {
  return parseResponse<ExperimentResult[]>(await fetch(`${API_URL}/api/experiments`));
}

export async function registerPerson(name: string, files: File[]) {
  const form = new FormData();
  form.append("name", name);
  files.forEach((file) => form.append("images", file));
  return parseResponse(await fetch(`${API_URL}/api/register`, { method: "POST", body: form }));
}

export async function analyzeImage(file: File) {
  const form = new FormData();
  form.append("image", file);
  return parseResponse<AnalyzeResponse>(await fetch(`${API_URL}/api/analyze`, { method: "POST", body: form }));
}

export async function recognizeImage(file: File) {
  const form = new FormData();
  form.append("image", file);
  return parseResponse<RecognizeResponse>(await fetch(`${API_URL}/api/recognize`, { method: "POST", body: form }));
}

# MolXAI API Documentation

## Overview

**Base URL:** `http://localhost:5000`  
**API Prefix:** `/api`  
**Content Type:** `application/json`  
**Version:** `1.0.0`

---

## Authentication

MolXAI uses **JWT (JSON Web Token)** authentication with a dual-token strategy:

- **Access Token** — Short-lived (15 min). Sent in every request as `Authorization: Bearer <token>`.
- **Refresh Token** — Long-lived (7 days). Stored as an HttpOnly, Secure, SameSite=Strict cookie. Used to issue new access tokens without re-login.

### Error Responses

| Status | Meaning |
|--------|---------|
| 400 | Bad Request — Validation error |
| 401 | Unauthorized — Missing or expired token |
| 403 | Forbidden — Insufficient role |
| 404 | Not Found |
| 409 | Conflict — Duplicate resource |
| 422 | Unprocessable Entity — Validation failure |
| 429 | Too Many Requests — Rate limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable — AI service offline |

---

## Standard Response Envelope

All responses follow this structure:

```json
{
  "success": true,
  "message": "Human-readable status",
  "data": {},
  "meta": {
    "total": 50,
    "page": 1,
    "limit": 10,
    "pages": 5
  },
  "errors": [
    { "field": "email", "message": "Invalid email address" }
  ]
}
```

---

## Endpoints

---

### 🔐 Authentication

#### POST `/api/auth/register`

Register a new user account.

**Request Body:**
```json
{
  "name": "Dr. Jane Smith",
  "email": "jane@example.com",
  "password": "SecurePass1"
}
```

**Password Requirements:** Min 8 chars, at least 1 uppercase, 1 lowercase, 1 digit.

**Response `201`:**
```json
{
  "success": true,
  "message": "Account created successfully",
  "data": {
    "user": { "id": "...", "name": "Dr. Jane Smith", "email": "jane@example.com", "role": "user" },
    "accessToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

---

#### POST `/api/auth/login`

Authenticate and receive tokens.

**Request Body:**
```json
{ "email": "jane@example.com", "password": "SecurePass1" }
```

**Response `200`:** Returns same structure as register. Also sets `refreshToken` HttpOnly cookie.

---

#### POST `/api/auth/logout`

Revoke the current session.

**Response `200`:** Clears the refresh token cookie.

---

#### POST `/api/auth/refresh`

Exchange the refresh token cookie for a new access token.

**Requires:** `refreshToken` cookie (sent automatically).

**Response `200`:**
```json
{ "data": { "accessToken": "new_access_token..." } }
```

---

### 👤 User

All user endpoints require: `Authorization: Bearer <access_token>`

#### GET `/api/user/profile`

Get the current user's profile and prediction count.

**Response `200`:**
```json
{
  "data": {
    "user": { "id": "...", "name": "Dr. Jane Smith", "email": "...", "role": "user" },
    "predictionCount": 42
  }
}
```

---

#### PUT `/api/user/profile`

Update profile name and/or email.

**Request Body:** `{ "name": "New Name" }` or `{ "email": "new@email.com" }`

---

#### PUT `/api/user/password`

Change user password. On success, all refresh tokens are revoked.

**Request Body:**
```json
{
  "currentPassword": "OldPass1",
  "newPassword": "NewPass2",
  "confirmPassword": "NewPass2"
}
```

---

#### DELETE `/api/user`

Permanently delete the account and all associated predictions.

---

### 🔬 Predictions

#### POST `/api/predictions`

Submit a SMILES string for toxicity prediction.

**Request Body:**
```json
{ "smiles": "CC(=O)Oc1ccccc1C(=O)O" }
```

**Response `201`:**
```json
{
  "data": {
    "prediction": {
      "_id": "...",
      "smiles": "CC(=O)Oc1ccccc1C(=O)O",
      "prediction": "toxic",
      "probability": 0.7234,
      "confidence": 0.8891,
      "importantAtoms": [0, 2, 4],
      "importantBonds": [{ "atomA": 0, "atomB": 1, "weight": 0.72 }],
      "executionTime": 847,
      "createdAt": "2024-01-15T10:30:00Z"
    }
  }
}
```

**Rate Limit:** 10 requests/minute per user.

---

#### GET `/api/predictions`

Get prediction history with pagination, search, and sort.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `page` | number | 1 | Page number |
| `limit` | number | 10 | Results per page (max 50) |
| `search` | string | — | Search within SMILES strings |
| `sort` | string | `createdAt` | Sort field |
| `order` | `asc\|desc` | `desc` | Sort direction |

**Response `200`:** Array of predictions with `meta` pagination object.

---

#### GET `/api/predictions/stats`

Get aggregate statistics for the current user.

**Response `200`:**
```json
{
  "data": {
    "stats": {
      "total": 42,
      "toxic": 18,
      "nonToxic": 21,
      "pending": 3,
      "avgConfidence": 0.847,
      "avgExecutionTime": 720
    }
  }
}
```

---

#### GET `/api/predictions/:id`

Get a specific prediction by ID (must belong to current user).

---

#### DELETE `/api/predictions/:id`

Delete a specific prediction (must belong to current user).

---

### 🤖 AI Service

#### GET `/api/ai/status`

Get AI model and service health status (public, no auth).

**Response `200`:**
```json
{
  "data": {
    "service": { "status": "offline" },
    "model": {
      "version": "0.0.0-placeholder",
      "name": "EQ-KA-GCN",
      "status": "offline",
      "description": "AI model integration pending."
    }
  }
}
```

---

#### POST `/api/ai/predict`

Direct proxy to the EQ-KA-GCN prediction service.

**Request Body:** `{ "smiles": "..." }`

**Response:** Same as `POST /api/predictions` result data.

---

#### POST `/api/ai/explain`

Generate GNN attention-based explainability data.

**Request Body:**
```json
{ "smiles": "CC(=O)Oc1ccccc1C(=O)O", "predictionId": "optional_id" }
```

**Response `200`:**
```json
{
  "data": {
    "atomAttentions": [{ "atomIndex": 0, "weight": 0.8234 }],
    "bondAttentions": [{ "bondIndex": 0, "weight": 0.7123, "atoms": [0, 1] }],
    "saliencyMap": []
  }
}
```

---

## Rate Limits

| Endpoint Group | Window | Limit |
|---------------|--------|-------|
| Global API | 15 min | 100 requests |
| Auth endpoints | 15 min | 20 requests |
| Prediction endpoints | 1 min | 10 requests |

---

## AI Integration

The AI service placeholder endpoints are located at `http://localhost:8000`:

- `GET /health` — Service health check
- `POST /api/predict` — GNN toxicity prediction
- `POST /api/explain` — GNN attention explanation

See `ai/app/services/gnn_service.py` for integration instructions.

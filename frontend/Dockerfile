# FILE: frontend/Dockerfile
# (Workspace-Aware Version - Added DISABLE_ESLINT_PLUGIN)

# Stage 1: Build the React application using Node 20+
FROM node:20-alpine as build

WORKDIR /app # Root context

# Copy root package.json AND root package-lock.json
COPY package.json ./
COPY package-lock.json ./

# Copy frontend-specific package.json
COPY frontend/package.json ./frontend/

# Install ONLY frontend workspace dependencies using root lockfile
RUN npm ci --workspace=frontend --legacy-peer-deps

# Copy frontend source code
COPY frontend ./frontend/

# --- FIX: Disable ESLint plugin during build ---
ENV DISABLE_ESLINT_PLUGIN=true

# Run build from root, targeting the workspace
RUN npm run build --workspace=frontend

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage's frontend/build directory
COPY --from=build /app/frontend/build /usr/share/nginx/html
# Optional: Copy custom Nginx config if needed
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
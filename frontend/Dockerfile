# FILE: frontend/Dockerfile
# (Workspace-Aware Version - Build command run from root)

# Stage 1: Build the React application using Node 20+
FROM node:20-alpine as build

# Set base working directory (represents project root in this stage)
WORKDIR /app

# Copy the root package.json AND the root package-lock.json
COPY package.json ./
COPY package-lock.json ./

# Copy the package.json specific to the frontend workspace
COPY frontend/package.json ./frontend/

# Install ONLY the frontend workspace dependencies using the root lock file
RUN npm ci --workspace=frontend --legacy-peer-deps

# Copy the rest of the frontend source code into its subdirectory
COPY frontend ./frontend/

# *** FIX: Run the build script from the ROOT (/app) directory, targeting the frontend workspace ***
# This ensures scripts can resolve dependencies hoisted to the root node_modules
RUN npm run build --workspace=frontend

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage's frontend/build directory
# The build output is still expected inside /app/frontend/build
COPY --from=build /app/frontend/build /usr/share/nginx/html
# Optional: Copy custom Nginx config if needed
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
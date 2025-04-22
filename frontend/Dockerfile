# FILE: frontend/Dockerfile
# (Corrected for npm workspaces - uses root lockfile)

# Stage 1: Build the React application using Node 20+
# Use Node 20+ as required by react-router
FROM node:20-alpine as build

# Set base working directory to represent project root in build stage
WORKDIR /app

# Copy the root package.json AND the root package-lock.json
# These define the workspaces and lock ALL dependencies
COPY package.json ./
COPY package-lock.json ./

# Copy the package.json specific to the frontend workspace
# This helps npm identify the workspace dependencies needed
COPY frontend/package.json ./frontend/

# Install ONLY the frontend workspace dependencies using the root lock file
# Use the --workspace flag. Add --include-workspace-root if needed.
RUN npm ci --workspace=frontend --legacy-peer-deps

# Copy the rest of the frontend source code into its subdirectory
COPY frontend ./frontend/

# Set the working directory to the frontend code for the build step
WORKDIR /app/frontend

# Run the build script defined in frontend/package.json
RUN npm run build

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage's frontend/build directory
COPY --from=build /app/frontend/build /usr/share/nginx/html
# Optional: Copy custom Nginx config if needed for routing
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
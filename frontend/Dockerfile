# FILE: frontend/Dockerfile
# (Workspace-Aware - Copying root node_modules into frontend dir before build)

# Stage 1: Build the React application using Node 20+
FROM node:20-alpine as build

WORKDIR /app # Root context

# Copy the root package.json AND the root package-lock.json
COPY package.json ./
COPY package-lock.json ./

# Install ALL dependencies using the root lockfile into /app/node_modules
RUN npm ci --legacy-peer-deps

# Copy the frontend source code into its subdirectory
COPY frontend ./frontend/

# *** FIX: Copy the installed root node_modules INTO the frontend directory ***
# This makes dependencies available relative to the frontend directory
RUN cp -a /app/node_modules ./frontend/

# Set the working directory to the frontend code
WORKDIR /app/frontend

# Run the build script (using the script defined in frontend/package.json)
# It should now find react-scripts in ./node_modules/.bin relative to this WORKDIR
RUN npm run build

# Stage 2: Serve the build output with Nginx
FROM nginx:1.25-alpine
# Copy built assets from the build stage's frontend/build directory
COPY --from=build /app/frontend/build /usr/share/nginx/html
# Optional: Copy custom Nginx config if needed
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
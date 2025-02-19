// backend/server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const routes = require('./routes');  // Import our routes (we’ll create this next)

const app = express();
const PORT = 5000;  // Our backend will run on port 5000

// Middleware setup:
// express.json() parses incoming JSON requests so we can access req.body.
// cors() allows our frontend (on a different port) to communicate with this server.
app.use(express.json());
app.use(cors());

// Connect to MongoDB:
// Adjust the connection string as necessary for your local MongoDB installation.
mongoose.connect('mongodb://localhost:27017/rag-app', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => {
  console.log("Connected to MongoDB");
}).catch(err => console.error("MongoDB connection error:", err));

// Use our defined routes for handling API endpoints.
app.use('/api', routes);

// Start the server:
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

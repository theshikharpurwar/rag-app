const express = require('express');
const { Qwen2_5_VLForConditionalGeneration, AutoProcessor } = require('transformers');
const { ColPali, ColPaliProcessor } = require('colpali-engine');
const { QdrantClient, models } = require('qdrant-client');

require('dotenv').config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

// Initialize Qwen2.5-VL model and processor
const modelPath = "Qwen/Qwen2.5-VL-3B-Instruct";
const model = Qwen2_5_VLForConditionalGeneration.from_pretrained(modelPath, {
  torch_dtype: 'bfloat16',
  attn_implementation: 'flash_attention_2',
  device_map: 'auto',
  cache_dir: './cache',
});
const processor = AutoProcessor.from_pretrained(modelPath, {
  cache_dir: './cache',
});

// Initialize ColPali model and processor
const colpaliModelName = "vidore/colpali-v1.2";
const colpaliEmbedModel = ColPali.from_pretrained(colpaliModelName);
const colpaliProcessor = ColPaliProcessor.from_pretrained(colpaliModelName);

// Initialize Qdrant client
const client = new QdrantClient({
  url: 'http://localhost:6333',
});

const collectionName = "deepseek-colpali-multimodalRAG";

// Create collection if it doesn't exist
async function createCollection() {
  if (!await client.collection_exists(collectionName)) {
    await client.create_collection({
      collection_name: collectionName,
      on_disk_payload: true,
      vectors_config: models.VectorParams.size(128).distance(models.Distance.COSINE).on_disk(true),
      multivector_config: models.MultiVectorConfig.comparator(models.MultiVectorComparator.MAX_SIM),
    });
  }
}

createCollection();

// API endpoint to handle queries
app.post('/api/query', async (req, res) => {
  const { query, image } = req.body;

  try {
    // Embed query text
    const queryEmbedding = await colpaliEmbedModel.embed_text(query);

    // Search for similar image embeddings
    const searchResult = await client.search(collectionName, queryEmbedding, {
      limit: 1,
    });

    if (searchResult_hits.length === 0) {
      return res.status(404).json({ message: 'No relevant images found.' });
    }

    const imageId = searchResult.hits[0].id;
    const imagePath = searchResult.hits[0].payload.image;

    // Prepare input for Qwen2.5-VL model
    const inputText = `Answer the question: ${query} based on the following image: ${imagePath}`;
    const inputs = processor(inputText, return_tensors="pt");

    // Generate response using Qwen2.5-VL model
    const output = await model.generate(**inputs);
    const response = processor.decode(output[0], skip_special_tokens=True);

    res.json({ response });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Internal Server Error' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
def store_embeddings(embeddings, collection_name, pdf_name, image_paths):
    """Store embeddings in Qdrant"""
    try:
        # Connect to Qdrant
        client = QdrantClient(url="http://localhost:6333")

        # Check if collection exists, if not create it
        try:
            collection_info = client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} exists")
        except Exception as e:
            if "doesn't exist" in str(e):
                logger.info(f"Creating collection {collection_name}")
                # Get vector size from the first embedding
                vector_size = len(embeddings[0])
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name} with vector size {vector_size}")
            else:
                raise

        # Prepare points for insertion
        points = []
        for i, (embedding, img_path) in enumerate(zip(embeddings, image_paths)):
            # Convert embedding to list if it's a numpy array or tensor
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy().tolist()

            # Use integer ID instead of string ID
            point_id = i + 1000000  # Start from a large number to avoid conflicts

            points.append(models.PointStruct(
                id=point_id,  # Use integer ID
                vector=embedding,
                payload={
                    "pdf_name": pdf_name,
                    "page_num": i + 1,
                    "image_path": img_path,
                    "original_id": f"{pdf_name}_{i}"  # Store the original ID in payload
                }
            ))

        # Insert points into collection
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Stored {len(points)} embeddings in collection {collection_name}")
        return len(points)
    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        raise
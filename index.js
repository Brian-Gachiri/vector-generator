import fs from "fs/promises";
import path from "path";
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from "openai"
import 'dotenv/config'

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;

if (!OPENAI_API_KEY || !PINECONE_API_KEY) {
    throw new Error("Kindly set the OPENAI_API_KEY and PINECONE_API_KEY in the environment variables.");
}

const openai = new OpenAI({
    apiKey: OPENAI_API_KEY
});

// Initialize Pinecone
const pinecone = new Pinecone({
    apiKey: PINECONE_API_KEY,
});

await pinecone.createIndex({
    name: process.env.PINECONE_INDEX || 'markdown-data',
    dimension: 1536, // Replace with your model dimensions
    metric: 'cosine', // Replace with your model metric
    spec: { 
      serverless: { 
        cloud: 'aws', 
        region: 'us-east-1' 
      }
    } 
  });
const index = pinecone.index(process.env.PINECONE_INDEX || 'markdown-data');

const BATCH_SIZE = 10; // Number of vectors to upsert in a single batch
const MAX_PARALLEL_REQUESTS = 5; // Limit concurrent requests to avoid overwhelming APIs

/**
 * Recursively finds all Markdown files in a directory.
 * @param dirPath - Directory path to search.
 * @returns An array of file paths.
 */
async function findMarkdownFiles(dirPath) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const files = await Promise.all(
    entries.map((entry) => {
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        return findMarkdownFiles(fullPath);
      } else if (entry.isFile() && entry.name.endsWith(".md")) {
        return fullPath;
      } else {
        return [];
      }
    })
  );
  return files.flat();
}

/**
 * Embeds data using OpenAI's embedding model.
 * @param text - The text to embed.
 * @returns A Promise resolving to the embedding vector.
 */
async function generateEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: text,
  });
  console.log(response.data[0])
  return response.data[0].embedding;
}

/**
 * Saves a batch of embeddings to Pinecone.
 * @param vectors - Array of vectors to upsert.
 */
async function batchSaveToPinecone(vectors) {
  await index.upsert(vectors);
}

/**
 * Processes Markdown files in parallel, generates embeddings, and batches them for Pinecone.
 * @param markdownFiles - Array of file paths to process.
 */
async function processFilesInParallel(markdownFiles) {
  const chunks = chunkArray(markdownFiles, MAX_PARALLEL_REQUESTS);

  for (const chunk of chunks) {
    await Promise.all(
      chunk.map(async (filePath) => {
        try {
          const content = await fs.readFile(filePath, "utf-8");
          const embedding = await generateEmbedding(content);
          const fileName = path.basename(filePath, path.extname(filePath));

          return {
            id: fileName,
            values: embedding,
            metadata: {
              filePath,
              fileName,
              folder: path.dirname(filePath),
            },
          };
        } catch (error) {
          console.error(`Failed to process file (${filePath}):`, error.message);
          return null;
        }
      })
    ).then((vectors) => {
      const validVectors = vectors.filter((v) => v !== null);

      if (validVectors.length > 0) {
        const batches = chunkArray(validVectors, BATCH_SIZE);
        batches.forEach(async (batch) => await batchSaveToPinecone(batch));
      }
    });
  }
}

/**
 * Chunks an array into smaller arrays of a specified size.
 * @param array - The array to chunk.
 * @param chunkSize - The size of each chunk.
 * @returns An array of chunks.
 */
function chunkArray(array, chunkSize) {
  const chunks = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    chunks.push(array.slice(i, i + chunkSize));
  }
  return chunks;
}

/**
 * Processes a folder to find Markdown files, generate embeddings, and save them to Pinecone.
 * @param folderPath - Path to the folder to process.
 */
async function processMarkdownFolder(folderPath) {
  const markdownFiles = await findMarkdownFiles(folderPath);

  console.log(`Found ${markdownFiles.length} Markdown files.`);

  if (markdownFiles.length > 0) {
    await processFilesInParallel(markdownFiles);
    console.log("Folder processing complete!");
  } else {
    console.log("No Markdown files found.");
  }
}

// Example usage
const folderToProcess = "/path/to/data";
processMarkdownFolder(folderToProcess)
  .then(() => console.log("All tasks completed successfully!"))
  .catch((error) => console.error("Error processing folder:", error.message));

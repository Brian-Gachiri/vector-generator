import fs from "fs/promises";
import path from "path";
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from "openai";
import tokenizer from "gpt-tokenizer";
import 'dotenv/config';

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

pinecone.listIndexes().then(async(response) => {
    if (!response.indexes.some((index) => [process.env.PINECONE_INDEX, 'markdown-data'].includes(index.name))) {
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
    }
})

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
 * Calculates the token count of a given text using gpt-tokenizer.
 * @param text - The text to calculate tokens for.
 * @returns The number of tokens in the text.
 */
function getTokenCount(text) {
  return tokenizer.encode(text).length;
}

/**
 * Splits a text into chunks that fit within the token limit.
 * Includes multiple fallback mechanisms to prevent infinite loops.
 * @param text - The text to split.
 * @param maxTokens - Maximum tokens allowed per chunk.
 * @returns An array of text chunks.
 */
function splitTextIntoChunks(text, maxTokens = 8000) {
  const chunks = [];
  const paragraphs = text.split(/\n\n+/); 
  let currentChunk = "";

  for (const paragraph of paragraphs) {
    const paragraphTokens = getTokenCount(paragraph);

    // Safety check: Prevent infinite loops if a paragraph is excessively large
    if (paragraphTokens > maxTokens * 10) { 
      console.warn(`Extremely large paragraph detected. Attempting fallback splitting.`);
      const sentenceChunks = splitBySentences(paragraph, maxTokens); 
      chunks.push(...sentenceChunks);
      continue; 
    }

    if (paragraphTokens > maxTokens) {
      const sentenceChunks = splitBySentences(paragraph, maxTokens);
      chunks.push(...sentenceChunks);
    } else {
      if (getTokenCount(currentChunk) + paragraphTokens > maxTokens) {
        chunks.push(currentChunk);
        currentChunk = paragraph;
      } else {
        currentChunk += `${currentChunk ? "\n\n" : ""}${paragraph}`;
      }
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk);
  }

  return chunks;
}

/**
 * Splits text into chunks by sentences.
 * @param text - The text to split.
 * @param maxTokens - Maximum tokens allowed per chunk.
 * @returns An array of text chunks.
 */
function splitBySentences(text, maxTokens) {
  const sentences = text.split(/(?<=\.)\s+/); 
  const sentenceChunks = [];
  let currentChunk = "";

  for (const sentence of sentences) {
    const sentenceTokens = getTokenCount(sentence);

    if (sentenceTokens > maxTokens) {
      console.warn(`Large sentence detected. Attempting character-based splitting.`);
      const charChunks = chunkByCharacters(sentence, maxTokens);
      sentenceChunks.push(...charChunks);
    } else {
      if (getTokenCount(currentChunk) + sentenceTokens > maxTokens) {
        sentenceChunks.push(currentChunk);
        currentChunk = sentence;
      } else {
        currentChunk += `${currentChunk ? " " : ""}${sentence}`; 
      }
    }
  }

  if (currentChunk) {
    sentenceChunks.push(currentChunk);
  }

  return sentenceChunks;
}

  
  /**
   * Splits text into chunks based on character limits as a fallback.
   * @param text - The text to split.
   * @param maxTokens - Maximum tokens allowed.
   * @returns An array of text chunks.
   */
  function chunkByCharacters(text, maxTokens) {
    const chunks = [];
    const maxChars = maxTokens * 4; // Approximate character limit (based on tokenization ratios)
    for (let i = 0; i < text.length; i += maxChars) {
      chunks.push(text.slice(i, i + maxChars));
    }
    return chunks;
  }
  
  
  /**
   * Logs the token count of each chunk for debugging.
   * @param chunks - Array of text chunks.
   */
  function logChunkDetails(chunks) {
    chunks.forEach((chunk, index) => {
      console.log(`Chunk ${index + 1}: ${getTokenCount(chunk)} tokens`);
    });
  }
/**
 * Embeds data using OpenAI's embedding model, handling large files by chunking.
 * @param text - The text to embed.
 * @returns A Promise resolving to an array of embedding vectors.
 */
async function generateEmbeddingsForLargeText(text) {
  const chunks = splitTextIntoChunks(text, 8000); // Ensure chunks fit within the model's context limit
  logChunkDetails(chunks);
  const data = [];

  for (const chunk of chunks) {
    const embedding = await generateEmbedding(chunk);
    data.push({
      embeddings: embedding,
      text: chunk
    });
  }

  return data;
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
  console.log(response.data[0]);
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

  if (markdownFiles.length === 0) {
    console.log("No Markdown files found.");
    return;
  }

  // Process files in parallel with error handling
  await Promise.allSettled(
    chunkArray(markdownFiles, MAX_PARALLEL_REQUESTS).map(async (chunk) => {
      for (const filePath of chunk) {
        try {
          const content = await fs.readFile(filePath, "utf-8");
          const embeddings = await generateEmbeddingsForLargeText(content);
          const fileName = path.basename(filePath, path.extname(filePath));

          const vectors = embeddings.map((embedding, index) => ({
            id: `${fileName}-${index}`,
            values: embedding.embeddings,
            metadata: {
              filePath,
              fileName,
              folder: path.dirname(filePath),
              chunkIndex: index,
              text: embedding.text
            },
          }));

          await batchSaveToPinecone(vectors);
        } catch (error) {
          console.error(`Failed to process file (${filePath}):`, error.message);
        }
      }
    })
  );

  console.log("Folder processing complete!");
}

// Example usage
const folderToProcess = "C:/Users/Gachiri/Documents/outputData";
processMarkdownFolder(folderToProcess)
  .then(() => console.log("All tasks completed successfully!"))
  .catch((error) => console.error("Error processing folder:", error.message));

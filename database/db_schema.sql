-- Table for raw jewelry items
CREATE TABLE IF NOT EXISTS jewelry_items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    price VARCHAR(50),
    url TEXT,
    image_url TEXT,
    condition VARCHAR(50),
    seller_reputation VARCHAR(100),
    auction_duration VARCHAR(50),
    material VARCHAR(100),
    color VARCHAR(50),
    brand VARCHAR(100),
    stone VARCHAR(100),
    stone_color VARCHAR(50),
    weight FLOAT,
    metal_purity VARCHAR(50),
    origin VARCHAR(100),
    style VARCHAR(100),
    date_made DATE,
    unisex BOOLEAN
);

-- Table for preprocessed images
CREATE TABLE IF NOT EXISTS preprocessed_images (
    id SERIAL PRIMARY KEY,
    original_image_url TEXT,
    processed_image BYTEA,
    attributes JSONB,
    augmentation_history JSONB
);

-- Table for model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    original_image_url TEXT,
    attributes JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for product listings
CREATE TABLE IF NOT EXISTS product_listings (
    id SERIAL PRIMARY KEY,
    listing_data JSONB,
    price FLOAT,
    ebay_response JSONB,
    etsy_response JSONB,
    shopify_response JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for price model performance
CREATE TABLE IF NOT EXISTS price_model_performance (
    id SERIAL PRIMARY KEY,
    mae FLOAT,
    rmse FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for model evaluation metrics
CREATE TABLE IF NOT EXISTS model_evaluation (
    id SERIAL PRIMARY KEY,
    attribute VARCHAR(100),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

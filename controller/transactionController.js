const express = require("express");
const router = express.Router();
const fs = require("fs");
const path = require("path");
const bayes = require("bayes");
const Validator = require("fastest-validator");
const v = new Validator();

// Models
const Transaction = require("../model/Transaction");
const Saldo = require("../model/Saldo");
const Category = require("../model/Category");

// Middleware & Utils
const ErrorHandler = require("../utils/ErrorHandler");
const catchAsyncErrors = require("../middleware/catchAsyncErrors");
const { isAuthenticated } = require("../middleware/auth");

// Config
const MODEL_PATH = path.join(__dirname, "../model.json");
const BACKUP_MODEL_PATH = path.join(__dirname, "../model_backup.json");

// Initialize Bayes Classifier
let classifier = bayes();
let isModelReady = false;
let lastTrainingDate = null;

// ========================
//  🧠 MODEL HANDLING
// ========================

/**
 * Generate confusion matrix from test data
 */
async function generateConfusionMatrix(testData) {
    const matrix = {};
    const categories = new Set();

    // Initialize matrix
    testData.forEach(item => {
        categories.add(item.actualCategory);
        categories.add(item.predictedCategory);
    });

    Array.from(categories).forEach(cat => {
        matrix[cat] = {};
        Array.from(categories).forEach(otherCat => {
            matrix[cat][otherCat] = 0;
        });
    });

    // Populate matrix
    testData.forEach(item => {
        matrix[item.actualCategory][item.predictedCategory]++;
    });

    return matrix;
}

/**
 * Calculate evaluation metrics from confusion matrix
 */
function calculateMetrics(matrix) {
    const metrics = {};
    const categories = Object.keys(matrix);

    categories.forEach(category => {
        const truePositives = matrix[category][category];
        let falsePositives = 0;
        let falseNegatives = 0;

        categories.forEach(otherCategory => {
            if (otherCategory !== category) {
                falsePositives += matrix[otherCategory][category] || 0;
                falseNegatives += matrix[category][otherCategory] || 0;
            }
        });

        const precision = truePositives / (truePositives + falsePositives) || 0;
        const recall = truePositives / (truePositives + falseNegatives) || 0;
        const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

        metrics[category] = {
            precision: parseFloat(precision.toFixed(4)),
            recall: parseFloat(recall.toFixed(4)),
            f1Score: parseFloat(f1Score.toFixed(4)),
            support: truePositives + falseNegatives
        };
    });

    return metrics;
}

/**
 * @route   POST /api/transactions/evaluate-model
 * @desc    Evaluate model performance with confusion matrix
 */
router.post(
    "/evaluate-model",
    isAuthenticated,
    catchAsyncErrors(async (req, res) => {
        try {
            // Split data into training and testing (80/20)
            const allTransactions = await Transaction.find()
                .populate("category")
                .limit(1000); // Limit for performance

            if (allTransactions.length < 10) {
                return res.status(400).json({
                    code: 400,
                    status: "error",
                    message: "Not enough data for evaluation (min 10 transactions)"
                });
            }

            const splitIndex = Math.floor(allTransactions.length * 0.8);
            const trainingData = allTransactions.slice(0, splitIndex);
            const testData = allTransactions.slice(splitIndex);

            // Train temporary classifier
            const tempClassifier = bayes();

            trainingData.forEach(tx => {
                const inputText = `${tx.description} ${tx.type}`.toLowerCase();
                const categoryName = tx.category?.name || "unknown";
                tempClassifier.learn(inputText, categoryName);
            });

            // Test predictions
            const predictions = [];

            testData.forEach(tx => {
                const inputText = `${tx.description} ${tx.type}`.toLowerCase();
                const actualCategory = tx.category?.name || "unknown";
                const predictedCategory = tempClassifier.categorize(inputText);

                predictions.push({
                    description: tx.description,
                    type: tx.type,
                    actualCategory,
                    predictedCategory,
                    isCorrect: actualCategory === predictedCategory
                });
            });

            // Calculate metrics
            const accuracy = predictions.filter(p => p.isCorrect).length / predictions.length;
            const confusionMatrix = await generateConfusionMatrix(predictions);
            const metrics = calculateMetrics(confusionMatrix);

            return res.status(200).json({
                code: 200,
                status: "success",
                data: {
                    accuracy: parseFloat(accuracy.toFixed(4)),
                    totalTestSamples: predictions.length,
                    confusionMatrix,
                    metrics,
                    predictions: predictions.slice(0, 20) // Sample predictions
                }
            });

        } catch (err) {
            return res.status(500).json({
                code: 500,
                status: "error",
                message: "Model evaluation failed",
                error: err.message
            });
        }
    })
);

/**
 * @route   POST /api/transactions/cross-validate
 * @desc    Perform k-fold cross validation
 */
router.post(
    "/cross-validate",
    isAuthenticated,
    catchAsyncErrors(async (req, res) => {
        try {
            const k = req.body.k || 5; // Number of folds
            const transactions = await Transaction.find()
                .populate("category")
                .limit(500);

            if (transactions.length < k * 2) {
                return res.status(400).json({
                    code: 400,
                    status: "error",
                    message: `Not enough data for ${k}-fold cross validation`
                });
            }

            const foldSize = Math.floor(transactions.length / k);
            const results = [];

            for (let i = 0; i < k; i++) {
                const testStart = i * foldSize;
                const testEnd = (i + 1) * foldSize;
                const testData = transactions.slice(testStart, testEnd);
                const trainingData = [
                    ...transactions.slice(0, testStart),
                    ...transactions.slice(testEnd)
                ];

                const tempClassifier = bayes();

                // Train
                trainingData.forEach(tx => {
                    const inputText = `${tx.description} ${tx.type}`.toLowerCase();
                    const categoryName = tx.category?.name || "unknown";
                    tempClassifier.learn(inputText, categoryName);
                });

                // Test
                let correct = 0;
                testData.forEach(tx => {
                    const inputText = `${tx.description} ${tx.type}`.toLowerCase();
                    const actualCategory = tx.category?.name || "unknown";
                    const predictedCategory = tempClassifier.categorize(inputText);

                    if (actualCategory === predictedCategory) {
                        correct++;
                    }
                });

                const accuracy = correct / testData.length;
                results.push({
                    fold: i + 1,
                    accuracy: parseFloat(accuracy.toFixed(4)),
                    testSamples: testData.length
                });
            }

            const avgAccuracy = results.reduce((sum, result) => sum + result.accuracy, 0) / results.length;

            return res.status(200).json({
                code: 200,
                status: "success",
                data: {
                    k,
                    averageAccuracy: parseFloat(avgAccuracy.toFixed(4)),
                    folds: results,
                    totalSamples: transactions.length
                }
            });

        } catch (err) {
            return res.status(500).json({
                code: 500,
                status: "error",
                message: "Cross validation failed",
                error: err.message
            });
        }
    })
);

/**
 * Save the trained model to a JSON file with backup
 */
function saveModel() {
    try {
        const modelState = classifier.toJson();

        // Create backup if existing model exists
        if (fs.existsSync(MODEL_PATH)) {
            fs.copyFileSync(MODEL_PATH, BACKUP_MODEL_PATH);
            console.log("💾 Created model backup");
        }

        fs.writeFileSync(MODEL_PATH, modelState, "utf8");
        console.log("💾 Model saved successfully.");
        return true;
    } catch (err) {
        console.error("❌ Failed to save model:", err);
        return false;
    }
}

/**
 * Load the model from a JSON file (if exists) with fallback
 */
function loadModel() {
    try {
        if (fs.existsSync(MODEL_PATH)) {
            console.log("ℹ Model file found at:", MODEL_PATH);
            const modelData = fs.readFileSync(MODEL_PATH, "utf8");

            // Verify the file isn't empty
            if (!modelData.trim()) {
                throw new Error("Model file is empty");
            }

            // Try parsing to verify JSON is valid
            JSON.parse(modelData);

            // If we got here, JSON is valid
            classifier = bayes.fromJson(modelData);
            console.log("✅ Model loaded successfully.");
            isModelReady = true;
            return true;
        } else {
            console.log("ℹ No pre-trained model found at:", MODEL_PATH);
            // Initialize fresh classifier
            classifier = bayes();
            isModelReady = false;
            return false;
        }
    } catch (err) {
        console.error("❌ Failed to load model:", err);

        // Try loading from backup
        if (fs.existsSync(BACKUP_MODEL_PATH)) {
            console.log("🔄 Attempting to load from backup...");
            try {
                const backupData = fs.readFileSync(BACKUP_MODEL_PATH, "utf8");
                classifier = bayes.fromJson(backupData);
                console.log("✅ Backup model loaded successfully.");
                isModelReady = true;

                // Restore the backup
                fs.copyFileSync(BACKUP_MODEL_PATH, MODEL_PATH);
                console.log("🔄 Restored model from backup");
                return true;
            } catch (backupErr) {
                console.error("❌ Failed to load backup model:", backupErr);
            }
        }

        // Initialize fresh classifier if all fails
        console.log("🆕 Initializing fresh classifier...");
        classifier = bayes();
        isModelReady = false;

        // Clean up corrupt files
        if (fs.existsSync(MODEL_PATH)) {
            fs.unlinkSync(MODEL_PATH);
            console.log("🗑 Deleted corrupt model file");
        }
        return false;
    }
}

/**
 * Train the classifier using existing transactions
 */
async function trainClassifier() {
    try {
        console.log("🔁 Training model...");
        const startTime = Date.now();

        const batchSize = 100;
        let skip = 0;
        let trainedCount = 0;
        let categoriesUsed = new Set();

        while (true) {
            const transactions = await Transaction.find()
                .skip(skip)
                .limit(batchSize)
                .populate("category");

            if (transactions.length === 0) break;

            transactions.forEach((tx) => {
                const inputText = `${tx.description} ${tx.type}`.toLowerCase();
                const categoryName = tx.category?.name || "unknown";
                classifier.learn(inputText, categoryName);
                categoriesUsed.add(categoryName);
                trainedCount++;
            });

            skip += batchSize;
        }

        if (trainedCount > 0) {
            const success = saveModel();
            if (success) {
                lastTrainingDate = new Date();
                isModelReady = true;
                const duration = (Date.now() - startTime) / 1000;
                console.log(`🎉 Model training completed in ${duration}s`);
                console.log(`📊 Trained with ${trainedCount} transactions`);
                console.log(`🏷️ Categories used: ${Array.from(categoriesUsed).join(', ')}`);
                return true;
            }
        } else {
            console.log("ℹ No transactions found for training.");
        }
        return false;
    } catch (err) {
        console.error("❌ Model training failed:", err);
        isModelReady = false;
        return false;
    }
}

/**
 * Ensure the model is ready before making predictions
 */
async function ensureModelReady() {
    if (!isModelReady) {
        console.log("⚠ Model not ready - attempting to train...");
        return await trainClassifier();
    }
    return true;
}

/**
 * Predict category for a transaction
 */
async function predictCategory(description, type) {
    try {
        if (!isModelReady) {
            await ensureModelReady();
        }

        if (isModelReady) {
            const inputText = `${description} ${type}`.toLowerCase();
            const prediction = classifier.categorize(inputText);
            console.log(`🔮 Prediction: '${inputText}' → '${prediction}'`);
            return prediction || "uncategorized";
        }
        return "uncategorized";
    } catch (err) {
        console.error("❌ Prediction failed:", err);
        return "uncategorized";
    }
}

// Load model on startup
loadModel();

// ========================
//  🏦 MODELS ROUTES
// ========================

/**
 * @route   GET /api/transactions/model-status
 * @desc    Check the status of the ML model
 */
router.get(
    "/model-status",
    catchAsyncErrors(async (req, res) => {
        return res.status(200).json({
            code: 200,
            status: "success",
            data: {
                isModelReady,
                lastTrainingDate,
                modelPath: MODEL_PATH,
                modelExists: fs.existsSync(MODEL_PATH),
                backupExists: fs.existsSync(BACKUP_MODEL_PATH),
                classifierInfo: {
                    totalCategories: isModelReady ? classifier.categories.length : 0,
                },
            },
        });
    })
);

/**
 * @route   POST /api/transactions/train-model
 * @desc    Manually trigger model training
 */
router.post(
    "/train-model",
    isAuthenticated,
    catchAsyncErrors(async (req, res) => {
        const success = await trainClassifier();
        return res.status(200).json({
            code: 200,
            status: success ? "success" : "error",
            message: success
                ? "Model trained successfully"
                : "Model training failed",
            data: {
                isModelReady,
                lastTrainingDate,
                modelPath: MODEL_PATH,
            },
        });
    })
);

/**
 * @route   POST /api/transactions/reset-model
 * @desc    Reset the ML model
 */
router.post(
    "/reset-model",
    isAuthenticated,
    catchAsyncErrors(async (req, res) => {
        try {
            // Delete existing model files
            if (fs.existsSync(MODEL_PATH)) {
                fs.unlinkSync(MODEL_PATH);
            }
            if (fs.existsSync(BACKUP_MODEL_PATH)) {
                fs.unlinkSync(BACKUP_MODEL_PATH);
            }

            // Create fresh classifier
            classifier = bayes();
            isModelReady = false;
            lastTrainingDate = null;

            return res.status(200).json({
                code: 200,
                status: "success",
                message: "Model reset successfully",
                data: {
                    isModelReady,
                    lastTrainingDate,
                },
            });
        } catch (err) {
            return res.status(500).json({
                code: 500,
                status: "error",
                message: "Failed to reset model",
                error: err.message,
            });
        }
    })
);


// ========================
//  🏦 TRANSACTION ROUTES
// ========================

/**
 * @route   POST /api/transactions
 * @desc    Create a new transaction & update saldo
 */
router.post(
    "/models",
    isAuthenticated,
    catchAsyncErrors(async (req, res, next) => {
        // 1️⃣ Input Validation
        const schema = {
            user: { type: "string", empty: false },
            saldo: { type: "string", empty: false },
            amount: { type: "number", empty: false, convert: true },
            description: { type: "string", empty: false, max: 1024 },
            type: { type: "enum", values: ["income", "expense"] },
        };

        const validation = v.validate(req.body, schema);
        if (validation !== true) {
            return res.status(400).json({
                code: 400,
                status: "error",
                data: { error: "Validation failed", details: validation },
            });
        }

        const { user, saldo: saldoId, amount, description, type } = req.body;

        // 2️⃣ Check if Saldo exists
        const saldo = await Saldo.findById(saldoId);
        if (!saldo) {
            return res.status(404).json({
                code: 404,
                message: "Saldo not found",
            });
        }

        // 3️⃣ Predict Category
        const predictedCategoryName = await predictCategory(description, type);

        // 4️⃣ Find or Create Category (Case-Insensitive)
        let predictedCategory = await Category.findOne({
            name: { $regex: new RegExp(`^${predictedCategoryName}$`, "i") },
            type,
        });

        if (!predictedCategory) {
            predictedCategory = await Category.create({
                name: predictedCategoryName.toLowerCase(),
                type,
                description: `Auto-generated category for ${predictedCategoryName}`,
            });
        }

        // 5️⃣ Update Saldo (with validation)
        const newAmount = type === "income"
            ? saldo.amount + amount
            : saldo.amount - amount;

        if (newAmount < 0 && type === "expense") {
            return res.status(400).json({
                code: 400,
                status: "error",
                message: "Insufficient balance for this transaction",
            });
        }

        saldo.amount = newAmount;
        await saldo.save();

        // 6️⃣ Save Transaction
        const transaction = await Transaction.create({
            user,
            category: predictedCategory._id,
            saldo: saldoId,
            amount,
            description,
            type,
        });

        // 7️⃣ Return Response
        const resTransaction = await Transaction.findById(transaction._id)
            .populate("user", "name email")
            .populate("category", "name")
            .populate("saldo", "name amount");

        return res.status(200).json({
            code: 200,
            status: "success",
            data: {
                ...resTransaction.toObject(),
                predicted_category: predictedCategory.name,
                model_status: isModelReady ? "ready" : "not_ready",
            },
        });
    })
);

router.post(
    "",
    isAuthenticated,
    catchAsyncErrors(async (req, res, next) => {
        const schema = {
            user: { type: "string", empty: false },
            category: { type: "string", empty: false },
            saldo: { type: "string", empty: false },
            amount: { type: "number", empty: false, convert: true },
            description: { type: "string", empty: false, max: 1024 },
            type: { type: "enum", values: ["income", "expense"] },
        };

        const validation = v.validate(req.body, schema);
        if (validation !== true) {
            return res.status(400).json({
                code: 400,
                status: "error",
                data: { error: "Validation failed", details: validation },
            });
        }

        const { user, category, saldo: saldoId, amount, description, type } = req.body;

        const saldo = await Saldo.findById(saldoId);
        if (!saldo) {
            return res.status(404).json({
                code: 404,
                message: "Saldo not found",
            });
        }

        // Update saldo amount
        if (type === "income") {
            saldo.amount += amount;
        } else if (type === "expense") {
            saldo.amount -= amount;
        }

        await saldo.save();

        const transaction = await Transaction.create({
            user,
            category,
            saldo: saldoId,
            amount,
            description,
            type,
        });

        return res.status(200).json({
            code: 200,
            status: "success",
            data: transaction,
        });
    })
);

/**
 * @route   GET /transaction/list
 * @desc    Get all transactions
 */
router.get(
    "/list",
    isAuthenticated,
    catchAsyncErrors(async (req, res, next) => {
        const { page = 1, limit = 50 } = req.query;
        const skip = (Number(page) - 1) * Number(limit);

        const [items, total] = await Promise.all([
            Transaction.find()
                .populate("user", "name email")
                .populate("category", "name")
                .populate("saldo", "name amount")
                .sort({ createdAt: -1 })
                .skip(skip)
                .limit(Number(limit)),
            Transaction.countDocuments(),
        ]);

        res.status(200).json({
            meta: {
                message: "Transaction retrieved successfully",
                code: 200,
                status: "success",
                pagination: {
                    total,
                    page: Number(page),
                    limit: Number(limit),
                    pages: Math.ceil(total / Number(limit)),
                },
            },
            data: items,
        });
    })
);

/**
 * @route   GET /transaction/:id
 * @desc    Get transaction by ID
 */
router.get(
    "/:id",
    isAuthenticated,
    catchAsyncErrors(async (req, res, next) => {
        const transaction = await Transaction.findById(req.params.id)
            .populate("user", "name email")
            .populate("category", "name")
            .populate("saldo", "name amount");

        if (!transaction) {
            return res.status(404).json({
                code: 404,
                message: "Transaction not found",
            });
        }

        res.status(200).json({
            meta: {
                message: "Transaction retrieved successfully",
                code: 200,
                status: "success",
            },
            data: transaction,
        });
    })
);

/**
 * @route   PUT /transaction/:id
 * @desc    Update transaction by ID and update saldo accordingly
 */
router.put(
    "/:id",
    isAuthenticated,
    catchAsyncErrors(async (req, res, next) => {
        const schema = {
            category: { type: "string", optional: true },
            saldo: { type: "string", optional: true },
            amount: { type: "number", empty: false, convert: true, optional: true },
            description: { type: "string", empty: false, max: 1024, optional: true },
            type: { type: "enum", values: ["income", "expense"], optional: true },
        };

        const validation = v.validate(req.body, schema);
        if (validation !== true) {
            return res.status(400).json({
                code: 400,
                status: "error",
                data: { error: "Validation failed", details: validation },
            });
        }

        const existingTransaction = await Transaction.findById(req.params.id);
        if (!existingTransaction) {
            return res.status(404).json({
                code: 404,
                message: "Transaction not found",
            });
        }

        const saldo = await Saldo.findById(existingTransaction.saldo);
        if (!saldo) {
            return res.status(404).json({
                code: 404,
                message: "Original saldo not found",
            });
        }

        // Revert saldo change from old transaction
        if (existingTransaction.type === "income") {
            saldo.amount -= existingTransaction.amount;
        } else if (existingTransaction.type === "expense") {
            saldo.amount += existingTransaction.amount;
        }

        // Apply new values
        if (req.body.saldo && req.body.saldo !== existingTransaction.saldo.toString()) {
            // Jika saldo diubah, ambil saldo baru dan sesuaikan nilainya
            const newSaldo = await Saldo.findById(req.body.saldo);
            if (!newSaldo) {
                return res.status(404).json({
                    code: 404,
                    message: "New saldo not found",
                });
            }

            if (req.body.type === "income") {
                newSaldo.amount += req.body.amount;
            } else if (req.body.type === "expense") {
                newSaldo.amount -= req.body.amount;
            }
            await newSaldo.save();
        } else {
            // Jika saldo tetap, perbarui sesuai nilai baru
            if (req.body.type === "income") {
                saldo.amount += req.body.amount;
            } else if (req.body.type === "expense") {
                saldo.amount -= req.body.amount;
            }
            await saldo.save();
        }

        // Update transaction
        const updatedTransaction = await Transaction.findByIdAndUpdate(
            req.params.id,
            req.body,
            { new: true }
        );

        res.status(200).json({
            meta: {
                message: "Transaction updated successfully",
                code: 200,
                status: "success",
            },
            data: updatedTransaction,
        });
    })
);

/**
 * @route   DELETE /transaction/:id
 * @desc    Delete transaction by ID and revert saldo accordingly
 */
router.delete(
    "/:id",
    isAuthenticated,
    catchAsyncErrors(async (req, res, next) => {
        const transaction = await Transaction.findById(req.params.id);
        if (!transaction) {
            return res.status(404).json({
                code: 404,
                message: "Transaction not found",
            });
        }

        const saldo = await Saldo.findById(transaction.saldo);
        if (!saldo) {
            return res.status(404).json({
                code: 404,
                message: "Saldo not found",
            });
        }

        // Revert saldo sesuai tipe transaksi
        if (transaction.type === "income") {
            saldo.amount -= transaction.amount;
        } else if (transaction.type === "expense") {
            saldo.amount += transaction.amount;
        }
        await saldo.save();

        // Hapus transaksi
        await transaction.deleteOne();

        res.status(200).json({
            meta: {
                message: "Transaction deleted successfully",
                code: 200,
                status: "success",
            },
        });
    })
);


module.exports = router;

// ML Simulator - Simulates machine learning predictions and recommendations
const MLSimulator = {
    // Simulate collaborative filtering recommendations
    getCollaborativeRecommendations: function(userId = "demo_user", limit = 6) {
        const db = JSON.parse(localStorage.getItem('whiskeyDatabase'));
        const userRatings = db.userRatings.filter(r => r.userId === userId);
        const ratedIds = userRatings.map(r => r.whiskeyId);
        
        // Get user's average preferences
        const userPrefs = db.getAverageUserPreferences(userId);
        
        // Score all unrated whiskeys
        const recommendations = db.whiskeys
            .filter(w => !ratedIds.includes(w.id))
            .map(whiskey => {
                // Calculate similarity score based on flavor profile
                const similarity = this.calculateFlavorSimilarity(userPrefs, whiskey.flavorProfile);
                
                // Add collaborative filtering factor (simulated)
                const collaborativeScore = this.simulateCollaborativeScore(whiskey);
                
                // Combine scores
                const finalScore = (similarity * 0.7) + (collaborativeScore * 0.3);
                
                return {
                    ...whiskey,
                    recommendationScore: finalScore,
                    predictedRating: this.predictRating(userPrefs, whiskey),
                    matchPercentage: Math.round(finalScore * 100)
                };
            })
            .sort((a, b) => b.recommendationScore - a.recommendationScore)
            .slice(0, limit);
        
        return recommendations;
    },
    
    // Calculate flavor similarity between user preferences and whiskey
    calculateFlavorSimilarity: function(userPrefs, whiskeyProfile) {
        const flavorKeys = Object.keys(userPrefs);
        let totalDiff = 0;
        
        flavorKeys.forEach(flavor => {
            const diff = Math.abs(userPrefs[flavor] - whiskeyProfile[flavor]);
            totalDiff += diff;
        });
        
        // Normalize to 0-1 scale (max possible diff is 80)
        return 1 - (totalDiff / 80);
    },
    
    // Simulate collaborative filtering score
    simulateCollaborativeScore: function(whiskey) {
        // Use average rating and total ratings to simulate popularity
        const popularityScore = Math.min(whiskey.totalRatings / 5000, 1);
        const ratingScore = whiskey.averageRating / 5;
        
        // Add some randomness to simulate user-user similarities
        const randomFactor = 0.8 + (Math.random() * 0.4);
        
        return ((popularityScore * 0.3) + (ratingScore * 0.7)) * randomFactor;
    },
    
    // Predict user rating for a whiskey
    predictRating: function(userPrefs, whiskey) {
        const similarity = this.calculateFlavorSimilarity(userPrefs, whiskey.flavorProfile);
        
        // Base prediction on similarity and whiskey quality indicators
        const qualityScore = (whiskey.complexity + whiskey.viscosity) / 10;
        const priceAdjustment = whiskey.price > 50 ? 0.1 : 0;
        
        // Calculate predicted rating
        const baseRating = 2.5 + (similarity * 2) + (qualityScore * 0.5) + priceAdjustment;
        
        // Add some variance based on specific flavor preferences
        let adjustment = 0;
        if (userPrefs.sweet > 6 && whiskey.flavorProfile.sweet > 6) adjustment += 0.2;
        if (userPrefs.spicy > 6 && whiskey.flavorProfile.spicy > 6) adjustment += 0.2;
        if (userPrefs.raw > 6 && whiskey.flavorProfile.raw > 6) adjustment += 0.3;
        
        const finalRating = Math.min(5, Math.max(1, baseRating + adjustment));
        return Math.round(finalRating * 10) / 10;
    },
    
    // Get whiskeys matching a custom flavor profile
    getFlavorMatches: function(targetProfile, tastingNotes = [], limit = 8) {
        const db = JSON.parse(localStorage.getItem('whiskeyDatabase'));
        
        const matches = db.whiskeys
            .map(whiskey => {
                // Calculate profile match score
                const profileScore = this.calculateFlavorSimilarity(targetProfile, whiskey.flavorProfile);
                
                // Calculate tasting notes match score
                let notesScore = 0;
                if (tastingNotes.length > 0) {
                    const matchedNotes = whiskey.tastingNotes.filter(note => 
                        tastingNotes.some(target => note.toLowerCase().includes(target.toLowerCase()))
                    );
                    notesScore = matchedNotes.length / tastingNotes.length;
                }
                
                // Combine scores
                const finalScore = tastingNotes.length > 0 
                    ? (profileScore * 0.6) + (notesScore * 0.4)
                    : profileScore;
                
                return {
                    ...whiskey,
                    matchScore: finalScore,
                    matchPercentage: Math.round(finalScore * 100),
                    matchedNotes: whiskey.tastingNotes.filter(note => 
                        tastingNotes.some(target => note.toLowerCase().includes(target.toLowerCase()))
                    )
                };
            })
            .sort((a, b) => b.matchScore - a.matchScore)
            .slice(0, limit);
        
        return matches;
    },
    
    // Sensitivity analysis - determine which flavors influence ratings most
    getSensitivityAnalysis: function(userId = "demo_user") {
        const db = JSON.parse(localStorage.getItem('whiskeyDatabase'));
        const userRatings = db.userRatings.filter(r => r.userId === userId);
        
        // Calculate feature importance for each flavor dimension
        const flavorImportance = {
            sweet: 0, fruity: 0, floral: 0, spicy: 0,
            raw: 0, earthy: 0, dark: 0, light: 0
        };
        
        // Simulate SHAP values
        userRatings.forEach(rating => {
            const whiskey = db.getWhiskeyById(rating.whiskeyId);
            const ratingDiff = rating.rating - 3; // Difference from neutral
            
            Object.keys(flavorImportance).forEach(flavor => {
                // Simulate feature contribution
                const flavorStrength = whiskey.flavorProfile[flavor];
                const contribution = (flavorStrength - 5) * ratingDiff * 0.1;
                flavorImportance[flavor] += contribution;
            });
        });
        
        // Normalize and convert to percentages
        const total = Object.values(flavorImportance).reduce((sum, val) => sum + Math.abs(val), 0);
        Object.keys(flavorImportance).forEach(flavor => {
            flavorImportance[flavor] = {
                value: flavorImportance[flavor],
                percentage: Math.round((Math.abs(flavorImportance[flavor]) / total) * 100),
                direction: flavorImportance[flavor] > 0 ? 'positive' : 'negative'
            };
        });
        
        // Sort by importance
        const sortedImportance = Object.entries(flavorImportance)
            .sort((a, b) => Math.abs(b[1].value) - Math.abs(a[1].value))
            .map(([flavor, data]) => ({
                flavor,
                ...data
            }));
        
        // Generate insights
        const insights = this.generateInsights(sortedImportance, userRatings.length);
        
        return {
            flavorImportance: sortedImportance,
            insights,
            totalRatings: userRatings.length
        };
    },
    
    // Generate human-readable insights
    generateInsights: function(flavorImportance, totalRatings) {
        const insights = [];
        
        // Top positive influence
        const topPositive = flavorImportance.find(f => f.direction === 'positive');
        if (topPositive) {
            insights.push({
                type: 'positive',
                message: `You tend to rate whiskeys higher when they have strong ${topPositive.flavor} notes (${topPositive.percentage}% influence)`
            });
        }
        
        // Top negative influence
        const topNegative = flavorImportance.find(f => f.direction === 'negative');
        if (topNegative) {
            insights.push({
                type: 'negative',
                message: `You typically prefer whiskeys with less ${topNegative.flavor} character (${topNegative.percentage}% influence)`
            });
        }
        
        // Balance insight
        const balanced = flavorImportance.filter(f => f.percentage < 15 && f.percentage > 5);
        if (balanced.length >= 3) {
            insights.push({
                type: 'neutral',
                message: `You appreciate balanced whiskeys with moderate levels of multiple flavor profiles`
            });
        }
        
        // Experience insight
        if (totalRatings >= 5) {
            insights.push({
                type: 'info',
                message: `Based on ${totalRatings} ratings, we have a good understanding of your preferences`
            });
        }
        
        return insights;
    },
    
    // Get gift recommendations for another user
    getGiftRecommendations: function(recipientId, priceRange = null) {
        const db = JSON.parse(localStorage.getItem('whiskeyDatabase'));
        const recipient = db.giftRecipients.find(r => r.id === recipientId);
        
        if (!recipient) return [];
        
        // Filter by price range if specified
        let candidates = db.whiskeys;
        if (priceRange) {
            candidates = candidates.filter(w => w.priceCategory === priceRange);
        }
        
        // Score whiskeys based on recipient preferences
        const recommendations = candidates
            .map(whiskey => {
                // Type preference score
                const typeScore = recipient.preferredTypes.includes(whiskey.type) ? 0.3 : 0;
                
                // Flavor profile match
                const flavorScore = this.calculateFlavorSimilarity(
                    recipient.flavorPreferences, 
                    whiskey.flavorProfile
                ) * 0.7;
                
                // Check if already rated
                const alreadyRated = recipient.ratedWhiskeys.includes(whiskey.id);
                
                return {
                    ...whiskey,
                    giftScore: typeScore + flavorScore,
                    giftMatchPercentage: Math.round((typeScore + flavorScore) * 100),
                    alreadyOwned: alreadyRated,
                    reasoning: this.generateGiftReasoning(whiskey, recipient)
                };
            })
            .filter(w => !w.alreadyOwned)
            .sort((a, b) => b.giftScore - a.giftScore)
            .slice(0, 6);
        
        return recommendations;
    },
    
    // Generate gift reasoning
    generateGiftReasoning: function(whiskey, recipient) {
        const reasons = [];
        
        if (recipient.preferredTypes.includes(whiskey.type)) {
            reasons.push(`Matches their preference for ${whiskey.type}`);
        }
        
        // Check flavor preferences
        const topFlavors = Object.entries(recipient.flavorPreferences)
            .filter(([_, value]) => value >= 7)
            .map(([flavor, _]) => flavor);
        
        const matchingFlavors = topFlavors.filter(flavor => 
            whiskey.flavorProfile[flavor] >= 6
        );
        
        if (matchingFlavors.length > 0) {
            reasons.push(`Strong ${matchingFlavors.join(' and ')} notes they enjoy`);
        }
        
        if (whiskey.priceCategory === recipient.priceRange) {
            reasons.push('Within their usual price range');
        }
        
        return reasons.join('. ');
    },
    
    // Simulate processing time for realism
    simulateProcessing: async function(minTime = 500, maxTime = 1500) {
        const delay = minTime + Math.random() * (maxTime - minTime);
        return new Promise(resolve => setTimeout(resolve, delay));
    }
};

// Initialize if not already done
if (typeof(Storage) !== "undefined" && !localStorage.getItem('mlSimulator')) {
    localStorage.setItem('mlSimulator', 'initialized');
}
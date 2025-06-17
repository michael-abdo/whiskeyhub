// Whiskey Database - Mock data for ML demo
const whiskeyDatabase = {
    whiskeys: [
        {
            id: 1,
            name: "Glenfiddich 12 Year",
            brand: "Glenfiddich",
            type: "Single Malt Scotch",
            region: "Speyside",
            age: 12,
            proof: 80,
            price: 45,
            priceCategory: "$$",
            viscosity: 3,
            complexity: 3,
            flavorProfile: {
                sweet: 6,
                fruity: 8,
                floral: 5,
                spicy: 2,
                raw: 1,
                earthy: 3,
                dark: 2,
                light: 7
            },
            tastingNotes: ["pear", "apple", "honey", "vanilla", "oak"],
            distillersNotes: "Fresh and fruity with a hint of pear. Subtle oak notes with a delicate sweetness.",
            averageRating: 4.2,
            totalRatings: 3542,
            image: "glenfiddich-12.jpg"
        },
        {
            id: 2,
            name: "Buffalo Trace",
            brand: "Buffalo Trace",
            type: "Bourbon",
            region: "Kentucky",
            age: null,
            proof: 90,
            price: 30,
            priceCategory: "$",
            viscosity: 4,
            complexity: 4,
            flavorProfile: {
                sweet: 8,
                fruity: 4,
                floral: 2,
                spicy: 6,
                raw: 3,
                earthy: 5,
                dark: 6,
                light: 3
            },
            tastingNotes: ["caramel", "vanilla", "brown sugar", "cinnamon", "toffee"],
            distillersNotes: "Rich and complex with notes of caramel and vanilla. A touch of spice and oak.",
            averageRating: 4.4,
            totalRatings: 5123,
            image: "buffalo-trace.jpg"
        },
        {
            id: 3,
            name: "Jameson Irish Whiskey",
            brand: "Jameson",
            type: "Irish Whiskey",
            region: "Ireland",
            age: null,
            proof: 80,
            price: 28,
            priceCategory: "$",
            viscosity: 2,
            complexity: 2,
            flavorProfile: {
                sweet: 5,
                fruity: 4,
                floral: 6,
                spicy: 2,
                raw: 1,
                earthy: 2,
                dark: 1,
                light: 8
            },
            tastingNotes: ["green apple", "vanilla", "honey", "toasted wood"],
            distillersNotes: "Smooth and mellow with a perfect balance of spicy, nutty and vanilla notes.",
            averageRating: 3.9,
            totalRatings: 8234,
            image: "jameson.jpg"
        },
        {
            id: 4,
            name: "Lagavulin 16 Year",
            brand: "Lagavulin",
            type: "Single Malt Scotch",
            region: "Islay",
            age: 16,
            proof: 86,
            price: 100,
            priceCategory: "$$$",
            viscosity: 5,
            complexity: 5,
            flavorProfile: {
                sweet: 3,
                fruity: 2,
                floral: 1,
                spicy: 5,
                raw: 7,
                earthy: 9,
                dark: 9,
                light: 1
            },
            tastingNotes: ["peat smoke", "iodine", "seaweed", "dark chocolate", "leather"],
            distillersNotes: "Intensely flavored with peat smoke, iodine and seaweed. Rich and smoky.",
            averageRating: 4.6,
            totalRatings: 2876,
            image: "lagavulin-16.jpg"
        },
        {
            id: 5,
            name: "Maker's Mark",
            brand: "Maker's Mark",
            type: "Bourbon",
            region: "Kentucky",
            age: null,
            proof: 90,
            price: 32,
            priceCategory: "$",
            viscosity: 3,
            complexity: 3,
            flavorProfile: {
                sweet: 7,
                fruity: 5,
                floral: 3,
                spicy: 4,
                raw: 2,
                earthy: 4,
                dark: 5,
                light: 4
            },
            tastingNotes: ["caramel", "vanilla", "honey", "wheat", "butterscotch"],
            distillersNotes: "Smooth and subtle with notes of honey and fruit. Balanced with caramel and vanilla.",
            averageRating: 4.1,
            totalRatings: 6543,
            image: "makers-mark.jpg"
        },
        {
            id: 6,
            name: "Highland Park 12 Year",
            brand: "Highland Park",
            type: "Single Malt Scotch",
            region: "Orkney",
            age: 12,
            proof: 86,
            price: 55,
            priceCategory: "$$",
            viscosity: 4,
            complexity: 4,
            flavorProfile: {
                sweet: 5,
                fruity: 5,
                floral: 4,
                spicy: 4,
                raw: 5,
                earthy: 6,
                dark: 6,
                light: 3
            },
            tastingNotes: ["honey", "heather", "peat", "citrus", "smoke"],
            distillersNotes: "Balanced and harmonious with honey sweetness, heathery peat smoke and citrus.",
            averageRating: 4.3,
            totalRatings: 3987,
            image: "highland-park-12.jpg"
        },
        {
            id: 7,
            name: "Bulleit Rye",
            brand: "Bulleit",
            type: "Rye Whiskey",
            region: "Kentucky",
            age: null,
            proof: 90,
            price: 28,
            priceCategory: "$",
            viscosity: 3,
            complexity: 3,
            flavorProfile: {
                sweet: 3,
                fruity: 3,
                floral: 2,
                spicy: 9,
                raw: 4,
                earthy: 5,
                dark: 5,
                light: 3
            },
            tastingNotes: ["black pepper", "mint", "clove", "cinnamon", "cherry"],
            distillersNotes: "Spicy and bold with a distinct rye character. Notes of pepper and spice.",
            averageRating: 4.0,
            totalRatings: 4321,
            image: "bulleit-rye.jpg"
        },
        {
            id: 8,
            name: "Macallan 12 Year Double Cask",
            brand: "Macallan",
            type: "Single Malt Scotch",
            region: "Speyside",
            age: 12,
            proof: 86,
            price: 75,
            priceCategory: "$$$",
            viscosity: 4,
            complexity: 4,
            flavorProfile: {
                sweet: 8,
                fruity: 7,
                floral: 4,
                spicy: 3,
                raw: 1,
                earthy: 3,
                dark: 5,
                light: 4
            },
            tastingNotes: ["sherry", "vanilla", "butterscotch", "dried fruit", "ginger"],
            distillersNotes: "Rich and sweet with sherry influence. Notes of vanilla, butterscotch and spice.",
            averageRating: 4.5,
            totalRatings: 5678,
            image: "macallan-12.jpg"
        },
        {
            id: 9,
            name: "Wild Turkey 101",
            brand: "Wild Turkey",
            type: "Bourbon",
            region: "Kentucky",
            age: null,
            proof: 101,
            price: 25,
            priceCategory: "$",
            viscosity: 4,
            complexity: 4,
            flavorProfile: {
                sweet: 6,
                fruity: 3,
                floral: 1,
                spicy: 7,
                raw: 5,
                earthy: 6,
                dark: 7,
                light: 2
            },
            tastingNotes: ["caramel", "oak", "pepper", "tobacco", "vanilla"],
            distillersNotes: "Bold and spicy with a high proof kick. Rich caramel and vanilla notes.",
            averageRating: 4.2,
            totalRatings: 7890,
            image: "wild-turkey-101.jpg"
        },
        {
            id: 10,
            name: "Redbreast 12 Year",
            brand: "Redbreast",
            type: "Irish Whiskey",
            region: "Ireland",
            age: 12,
            proof: 80,
            price: 65,
            priceCategory: "$$",
            viscosity: 4,
            complexity: 4,
            flavorProfile: {
                sweet: 7,
                fruity: 8,
                floral: 5,
                spicy: 4,
                raw: 1,
                earthy: 3,
                dark: 3,
                light: 5
            },
            tastingNotes: ["sherry", "honey", "citrus", "toasted oak", "christmas spices"],
            distillersNotes: "Full-bodied with a perfect balance of spicy, creamy, fruity and sherry notes.",
            averageRating: 4.5,
            totalRatings: 3456,
            image: "redbreast-12.jpg"
        },
        {
            id: 11,
            name: "Ardbeg 10 Year",
            brand: "Ardbeg",
            type: "Single Malt Scotch",
            region: "Islay",
            age: 10,
            proof: 92,
            price: 55,
            priceCategory: "$$",
            viscosity: 4,
            complexity: 5,
            flavorProfile: {
                sweet: 2,
                fruity: 3,
                floral: 1,
                spicy: 4,
                raw: 8,
                earthy: 10,
                dark: 9,
                light: 1
            },
            tastingNotes: ["peat", "smoke", "tar", "lemon", "vanilla"],
            distillersNotes: "Intensely smoky with a surprising sweetness. Complex layers of peat and citrus.",
            averageRating: 4.4,
            totalRatings: 4567,
            image: "ardbeg-10.jpg"
        },
        {
            id: 12,
            name: "Knob Creek 9 Year",
            brand: "Knob Creek",
            type: "Bourbon",
            region: "Kentucky",
            age: 9,
            proof: 100,
            price: 35,
            priceCategory: "$",
            viscosity: 5,
            complexity: 4,
            flavorProfile: {
                sweet: 7,
                fruity: 4,
                floral: 2,
                spicy: 6,
                raw: 3,
                earthy: 5,
                dark: 7,
                light: 2
            },
            tastingNotes: ["maple", "vanilla", "toasted nuts", "caramel", "oak"],
            distillersNotes: "Full-bodied with rich maple and vanilla notes. Aged for depth and character.",
            averageRating: 4.3,
            totalRatings: 5432,
            image: "knob-creek-9.jpg"
        },
        {
            id: 13,
            name: "Glenlivet 12 Year",
            brand: "Glenlivet",
            type: "Single Malt Scotch",
            region: "Speyside",
            age: 12,
            proof: 80,
            price: 40,
            priceCategory: "$$",
            viscosity: 3,
            complexity: 3,
            flavorProfile: {
                sweet: 6,
                fruity: 7,
                floral: 6,
                spicy: 2,
                raw: 1,
                earthy: 2,
                dark: 2,
                light: 8
            },
            tastingNotes: ["apple", "pear", "floral", "honey", "almond"],
            distillersNotes: "Smooth and fruity with a delicate balance. Notes of summer fruits and florals.",
            averageRating: 4.1,
            totalRatings: 6789,
            image: "glenlivet-12.jpg"
        },
        {
            id: 14,
            name: "Four Roses Single Barrel",
            brand: "Four Roses",
            type: "Bourbon",
            region: "Kentucky",
            age: null,
            proof: 100,
            price: 45,
            priceCategory: "$$",
            viscosity: 4,
            complexity: 4,
            flavorProfile: {
                sweet: 7,
                fruity: 6,
                floral: 5,
                spicy: 5,
                raw: 2,
                earthy: 4,
                dark: 5,
                light: 4
            },
            tastingNotes: ["cherry", "vanilla", "oak", "cinnamon", "cocoa"],
            distillersNotes: "Complex and full-bodied with rich fruit and floral notes. Balanced with spice.",
            averageRating: 4.4,
            totalRatings: 4321,
            image: "four-roses-single.jpg"
        },
        {
            id: 15,
            name: "Talisker 10 Year",
            brand: "Talisker",
            type: "Single Malt Scotch",
            region: "Isle of Skye",
            age: 10,
            proof: 92,
            price: 65,
            priceCategory: "$$",
            viscosity: 4,
            complexity: 4,
            flavorProfile: {
                sweet: 3,
                fruity: 3,
                floral: 2,
                spicy: 7,
                raw: 6,
                earthy: 7,
                dark: 7,
                light: 2
            },
            tastingNotes: ["sea salt", "pepper", "smoke", "dried fruit", "maritime"],
            distillersNotes: "Maritime character with sea salt and smoke. Peppery spice with sweet undertones.",
            averageRating: 4.3,
            totalRatings: 3654,
            image: "talisker-10.jpg"
        }
    ],
    
    // User ratings history (simulated)
    userRatings: [
        { whiskeyId: 1, rating: 4.5, userId: "demo_user" },
        { whiskeyId: 2, rating: 4.8, userId: "demo_user" },
        { whiskeyId: 5, rating: 4.2, userId: "demo_user" },
        { whiskeyId: 8, rating: 4.7, userId: "demo_user" },
        { whiskeyId: 10, rating: 4.6, userId: "demo_user" },
        { whiskeyId: 13, rating: 4.0, userId: "demo_user" }
    ],
    
    // Gift recipient profiles (for gift guide feature)
    giftRecipients: [
        {
            id: "recipient_1",
            name: "John (Bourbon Lover)",
            preferredTypes: ["Bourbon", "Rye Whiskey"],
            flavorPreferences: {
                sweet: 7,
                fruity: 4,
                floral: 2,
                spicy: 6,
                raw: 3,
                earthy: 5,
                dark: 6,
                light: 3
            },
            priceRange: "$$",
            ratedWhiskeys: [2, 5, 7, 9, 12, 14]
        },
        {
            id: "recipient_2",
            name: "Sarah (Scotch Enthusiast)",
            preferredTypes: ["Single Malt Scotch"],
            flavorPreferences: {
                sweet: 4,
                fruity: 5,
                floral: 3,
                spicy: 5,
                raw: 6,
                earthy: 7,
                dark: 7,
                light: 2
            },
            priceRange: "$$$",
            ratedWhiskeys: [1, 4, 6, 8, 11, 15]
        },
        {
            id: "recipient_3",
            name: "Mike (Smooth & Sweet)",
            preferredTypes: ["Irish Whiskey", "Bourbon"],
            flavorPreferences: {
                sweet: 8,
                fruity: 6,
                floral: 5,
                spicy: 2,
                raw: 1,
                earthy: 2,
                dark: 2,
                light: 7
            },
            priceRange: "$",
            ratedWhiskeys: [3, 5, 10, 13]
        }
    ],
    
    // Flavor vocabulary for autocomplete
    flavorVocabulary: [
        "vanilla", "caramel", "honey", "maple", "butterscotch", "toffee",
        "apple", "pear", "citrus", "orange", "lemon", "cherry", "plum",
        "floral", "rose", "lavender", "heather",
        "pepper", "cinnamon", "clove", "ginger", "nutmeg",
        "smoke", "peat", "tobacco", "leather", "tar",
        "oak", "wood", "earth", "mineral", "grass",
        "chocolate", "coffee", "nuts", "almond",
        "light", "crisp", "fresh", "smooth", "delicate"
    ],
    
    // Helper functions
    getWhiskeyById: function(id) {
        return this.whiskeys.find(w => w.id === id);
    },
    
    getWhiskeysByType: function(type) {
        return this.whiskeys.filter(w => w.type === type);
    },
    
    getWhiskeysByPriceRange: function(priceCategory) {
        return this.whiskeys.filter(w => w.priceCategory === priceCategory);
    },
    
    getUserRatings: function(userId = "demo_user") {
        return this.userRatings.filter(r => r.userId === userId);
    },
    
    getAverageUserPreferences: function(userId = "demo_user") {
        const ratings = this.getUserRatings(userId);
        const ratedWhiskeys = ratings.map(r => this.getWhiskeyById(r.whiskeyId));
        
        const avgProfile = {
            sweet: 0, fruity: 0, floral: 0, spicy: 0,
            raw: 0, earthy: 0, dark: 0, light: 0
        };
        
        ratedWhiskeys.forEach(whiskey => {
            Object.keys(avgProfile).forEach(flavor => {
                avgProfile[flavor] += whiskey.flavorProfile[flavor] * 
                    (ratings.find(r => r.whiskeyId === whiskey.id).rating / 5);
            });
        });
        
        Object.keys(avgProfile).forEach(flavor => {
            avgProfile[flavor] = Math.round(avgProfile[flavor] / ratedWhiskeys.length);
        });
        
        return avgProfile;
    },
    
    searchWhiskeys: function(query) {
        const lowerQuery = query.toLowerCase();
        return this.whiskeys.filter(w => 
            w.name.toLowerCase().includes(lowerQuery) ||
            w.brand.toLowerCase().includes(lowerQuery) ||
            w.type.toLowerCase().includes(lowerQuery) ||
            w.tastingNotes.some(note => note.includes(lowerQuery))
        );
    }
};

// Store in localStorage for persistence across pages
if (typeof(Storage) !== "undefined") {
    localStorage.setItem('whiskeyDatabase', JSON.stringify(whiskeyDatabase));
}
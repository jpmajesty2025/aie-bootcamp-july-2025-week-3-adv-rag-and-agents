# 🔑 NVD API Key Setup Guide

## 📋 Prerequisites

You need an NVD API key to use the vulnerability system effectively. Here's how to set it up:

## 🚀 Step 1: Get Your NVD API Key

1. Visit the [NVD API Key Request Page](https://nvd.nist.gov/developers/request-an-api-key)
2. Fill out the form with your information
3. Check your email for the activation link
4. Click the link to activate your API key
5. **Save your API key securely** - you won't be able to retrieve it later

## 🔧 Step 2: Add API Key to Environment

### Option A: Add to .env file (Recommended)
Add this line to your `.env` file:
```
NVD_API_KEY=your_api_key_here
```

### Option B: Set Environment Variable
```bash
# Windows PowerShell
$env:NVD_API_KEY="your_api_key_here"

# Windows Command Prompt
set NVD_API_KEY=your_api_key_here

# Linux/Mac
export NVD_API_KEY="your_api_key_here"
```

## ✅ Step 3: Verify Setup

Run the vulnerability system:
```bash
python vulnerability_system.py
```

You should see:
- ✅ "This product uses data from the NVD API but is not endorsed or certified by the NVD."
- ✅ No more 404 errors from NVD API
- ✅ Faster scanning (50 requests per 30 seconds vs 5)

## 📊 Rate Limits

- **With API Key**: 50 requests per 30 seconds
- **Without API Key**: 5 requests per 30 seconds

## 🔒 Security Notes

- Keep your API key secure and don't commit it to version control
- The API key is case-sensitive
- Each email address can only have one active API key

## 🆘 Troubleshooting

If you still get errors:
1. Verify your API key is correct
2. Check that the environment variable is set
3. Restart your terminal/IDE after setting the environment variable
4. Ensure your API key is activated (check your email)

## 📚 Additional Resources

- [NVD API Documentation](https://nvd.nist.gov/developers/start-here)
- [API Key Request Page](https://nvd.nist.gov/developers/request-an-api-key)
- [Rate Limiting Information](https://nvd.nist.gov/developers/start-here#rate-limits) 
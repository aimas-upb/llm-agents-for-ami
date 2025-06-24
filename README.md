# Yggdrasil

A Virtual simulation of lab308 built with yggdrasil.

## Prerequisites

- Java 21 or higher
- Node.js and npm

## Running the project

### Backend

First, build the application. From the root of the project, run:
```bash
# On Linux/macOS
./gradlew

# On Windows
./gradlew.bat
```

Once the build is successful, you can start the backend server:
```bash
java -jar build/libs/yggdrasil-0.0.0-SNAPSHOT-all.jar -conf conf/cartago_config_light308.json
```

### Frontend

To run the frontend, first navigate to the `frontend` directory:
```bash
cd frontend
```

Install the dependencies:
```bash
npm install
```

Then, start the development server:
```bash
npm run dev
```

The frontend application will be available at `http://localhost:5173`.

import CopyWebpackPlugin from 'copy-webpack-plugin';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default {
    mode: 'development',
    devtool: 'source-map',
    entry: {
        'main': './main.js',
        'main.min': './main.js',
    },
    output: {
        filename: '[name].js',
        path: path.resolve(__dirname, 'dist'),
        library: {
            type: 'module',
        },
    },
    plugins: [
        // Copy .wasm files and index.html to dist folder
        new CopyWebpackPlugin({
            patterns: [
                {
                    from: 'node_modules/onnxruntime-web/dist/*.jsep.*',
                    to: '[name][ext]', // .wasmファイルをコピー
                },
                {
                    from: './index.html', // プロジェクトのルートにあるindex.htmlをコピー
                    to: 'index.html', // distディレクトリのルートにコピー
                },
                {
                    from: './main.css', // プロジェクトのルートにあるindex.htmlをコピー
                    to: 'main.css', // distディレクトリのルートにコピー
                },
            ],
        }),
    ],
    devServer: {
        static: {
            directory: path.resolve(__dirname, 'dist'),
        },
        port: 8080,
    },
    experiments: {
        outputModule: true,
    },
};
import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = document.location.pathname.replace('index.html', '') + 'dist/';


function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

//
// load file from server or cache
//
async function fetchAndCache(url) {
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = undefined; //await cache.match(url);
        if (cachedResponse === undefined) {
            log(`${url} (network)`);
            const buffer = await fetch(url).then(response => response.arrayBuffer());
            try {
                await cache.put(url, new Response(buffer));
            } catch (error) {
                console.error(error);
            }
            return buffer;
        }
        log(`${url} (cached)`);
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`can't fetch ${url}`);
        throw error;
    }
}

//
// class to handle a large language model on top of onnxruntime-web
//
export class LLM {
    sess_forward = undefined;
    sess_backward = undefined;
    profiler = false;
    feed_forward = {};
    feed_backward = {};
    output_tokens = [];
    need_position_ids = true;
    stop = false;
    kv_dims = [];
    dtype = "float16";
    max_tokens = 9999;

    constructor() {
    }

    async load(model_forward, options_forward, model_backward, options_backward) {
        ///////////////////
        const provider_forward = options_forward.provider || "webgpu";
        const local_forward = options_forward.local;
        const hasFP16_forward = (provider_forward === "wasm") ? false : options_forward.hasFP16;
        this.profiler = options_forward.profiler_forward;

        const model_path_forward = (local_forward) ? "models/" + model_forward.path : "https://huggingface.co/" + model_forward.path + "/resolve/main";
        let model_file_forward = model_forward.file || "model";
        model_file_forward = model_file_forward + ".onnx";

        log(`loading... ${model_forward.name},  ${provider_forward}`);
        const json_bytes_forward = await fetchAndCache(model_path_forward + "/config.json");
        let textDecoder = new TextDecoder();
        const model_config_forward = JSON.parse(textDecoder.decode(json_bytes_forward));

        const model_bytes_forward = await fetchAndCache(model_path_forward + "/onnx/" + model_file_forward);
        let modelSize_forward = model_bytes_forward.byteLength;
        log(`model size ${Math.round(modelSize_forward / 1024 / 1024)} MB`);

        const opt_forward = {
            executionProviders: [provider_forward],
            preferredOutputLocation: {},
        }

        switch (provider_forward) {
            case "webgpu":
                for (let i = 0; i < model_config_forward.num_hidden_layers; ++i) {
                    opt_forward.preferredOutputLocation[`present.${i}.key`] = 'gpu-buffer';
                    opt_forward.preferredOutputLocation[`present.${i}.value`] = 'gpu-buffer';
                }
                break;
        }

        ort.env.webgpu.profiling = {}
        if (this.profiler_forward) {
            opt.enableProfiling = true;
            ort.env.webgpu.profilingMode = 'default';
            ort.env.webgpu.profiling.mode = 'default';
        }

        this.sess_forward = await ort.InferenceSession.create(model_bytes_forward, opt_forward);
        ///////////////////////

        const provider_backward = options_backward.provider || "webgpu";
        const local_backward = options_backward.local;
        const hasFP16_backward = (provider_backward === "wasm") ? false : options_backward.hasFP16;

        const model_path_backward = (local_backward) ? "models/" + model_backward.path : "https://huggingface.co/" + model_backward.path + "/resolve/main";
        let model_file_backward = model_backward.file || "model";
        model_file_backward = model_file_backward + ".onnx";

        log(`loading... ${model_backward.name},  ${provider_backward}`);
        const json_bytes_backward = await fetchAndCache(model_path_backward + "/config.json");
        const model_config_backward = JSON.parse(textDecoder.decode(json_bytes_backward));

        const model_bytes_backward = await fetchAndCache(model_path_backward + "/onnx/" + model_file_backward);
        let modelSize_backward = model_bytes_backward.byteLength;
        log(`model size ${Math.round(modelSize_backward / 1024 / 1024)} MB`);

        const opt_backward = {
            executionProviders: [provider_backward],
            preferredOutputLocation: {},
        }

        switch (provider_backward) {
            case "webgpu":
                for (let i = 0; i < model_config_backward.num_hidden_layers; ++i) {
                    opt_backward.preferredOutputLocation[`present.${i}.key`] = 'gpu-buffer';
                    opt_backward.preferredOutputLocation[`present.${i}.value`] = 'gpu-buffer';
                }
                break;
        }

        ort.env.webgpu.profiling = {}

        this.sess_backward = await ort.InferenceSession.create(model_bytes_backward, opt_backward);
        ///////////////////////


        this.kv_dims = [1, model_config_forward.num_key_value_heads, 0, model_config_forward.hidden_size / model_config_forward.num_attention_heads];
        this.dtype = (hasFP16_forward) ? "float16" : "float32";
        this.num_layers = model_config_forward.num_hidden_layers;
        this.initilize_feed();
    }

    initilize_feed() {
        const feed_forward = this.feed_forward;
        const feed_backward = this.feed_backward;

        // dispose of previous gpu buffers
        for (const name in feed_forward) {
            const t = feed_forward[name];
            if (t.location === 'gpu-buffer') {
                t.dispose();
            }
        }
        for (const name in feed_backward) {
            const t = feed_backward[name];
            if (t.location === 'gpu-buffer') {
                t.dispose();
            }
        }
        this.feed_forward = {};
        this.feed_backward = {};
        // key value cache is zero copy, just pass gpu buffer as referece
        const empty = (this.dtype === "float16") ? new Uint16Array() : [];
        for (let i = 0; i < this.num_layers; ++i) {
            this.feed_forward[`past_key_values.${i}.key`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
            this.feed_forward[`past_key_values.${i}.value`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
        }
        for (let i = 0; i < this.num_layers; ++i) {
            this.feed_backward[`past_key_values.${i}.key`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
            this.feed_backward[`past_key_values.${i}.value`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
        }
        this.output_tokens = [];
    }

    //
    // poor mens argmax
    argmax(t, t2, block_ids) {
        const arr = t.data;
        const arr2 = t2.data;
        const start = t.dims[2] * (t.dims[1] - 1);
        let max = arr[start];
        let maxidx = 0;

        for (let i = 0; i < t.dims[2]; i++) {
            const val = arr[i + start] + arr2[i + start];
            if (!isFinite(val)) {
                throw new Error("found infinitive in logits");
            }
            if (val > max && !block_ids.includes(i)) {
                max = arr[i + start] + arr2[i + start];
                maxidx = i;
            }
        }
        return maxidx;
    }

    //
    // update key value cache
    //
    update_kv_cache(feed, outputs) {
        for (const name in outputs) {
            if (name.startsWith('present')) {
                let newName = name.replace('present', 'past_key_values');
                // dispose previous gpu buffers
                const t = feed[newName];
                if (t.location === 'gpu-buffer') {
                    t.dispose();
                }
                feed[newName] = outputs[name];
            }
        }
    }

    //
    // tell generate to stop()
    //
    abort() {
        this.stop = true;
    }

    // 
    // prefill prompt and generate tokens, greedy search only
    //


    async generate(tokens, generate_num, callback) {
        const max_tokens = generate_num;
        const feed_forward = this.feed_forward;
        const feed_backward = this.feed_backward;
        const input_ids_forward = new ort.Tensor('int64', BigInt64Array.from(([2].concat(tokens)).map(BigInt)), [1, tokens.length + 1]);
        const input_ids_backward = new ort.Tensor('int64', BigInt64Array.from(([5].concat(tokens)).map(BigInt)), [1, tokens.length + 1]);
        feed_forward['input_ids'] = input_ids_forward;
        feed_backward['input_ids'] = input_ids_backward;
        this.stop = false;

        this.output_tokens.push(...input_ids_forward.data);

        let last_token = 0;
        let seqlen = this.output_tokens.length;
        const input_len = input_ids_forward.size;

        if (this.need_position_ids) {
            feed_forward['position_ids'] = new ort.Tensor('int64', BigInt64Array.from({ length: input_len }, (_, i) => BigInt(seqlen - input_len + i)), [1, input_len]);
            feed_backward['position_ids'] = new ort.Tensor('int64', BigInt64Array.from({ length: input_len }, (_, i) => BigInt(seqlen - input_len + i)), [1, input_len]);
        }
        let tri_block_dict = new Map();
        while (seqlen < max_tokens && !this.stop) {
            seqlen = this.output_tokens.length;
            feed_forward['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen }, () => 1n), [1, seqlen]);
            feed_backward['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen }, () => 1n), [1, seqlen]);
            const outputs_forward = await this.sess_forward.run(feed_forward);
            const outputs_backward = await this.sess_backward.run(feed_backward);
            let block_ids = []
            if (seqlen >= 2 && tri_block_dict.has(`${this.output_tokens[seqlen - 2]},${this.output_tokens[seqlen - 1]}`)) {
                tri_block_dict.get(`${this.output_tokens[seqlen - 2]},${this.output_tokens[seqlen - 1]}`).forEach((elem) => block_ids.push(elem));
            }

            last_token = BigInt(this.argmax(outputs_forward.logits, outputs_backward.logits, block_ids));
            if (seqlen >= 2) {
                if (tri_block_dict.has(`${this.output_tokens[seqlen - 2]},${this.output_tokens[seqlen - 1]}`)) {
                    tri_block_dict.get(`${this.output_tokens[seqlen - 2]},${this.output_tokens[seqlen - 1]}`).push(Number(last_token));
                }
                else {
                    tri_block_dict.set(`${this.output_tokens[seqlen - 2]},${this.output_tokens[seqlen - 1]}`, [Number(last_token)]);
                }
            }

            this.output_tokens.push(last_token);
            if (callback && !this.profiler) {
                const temp_output = [...this.output_tokens, ...[...this.output_tokens].reverse().slice(1)];
                callback(temp_output);
            }
            this.update_kv_cache(feed_forward, outputs_forward);
            this.update_kv_cache(feed_backward, outputs_backward);
            feed_forward['input_ids'] = new ort.Tensor('int64', BigInt64Array.from(this.output_tokens), [1, seqlen + 1]);
            feed_backward['input_ids'] = new ort.Tensor('int64', BigInt64Array.from([BigInt(5), ...this.output_tokens.slice(1)]), [1, seqlen + 1]);
            if (this.need_position_ids) {
                feed_forward['position_ids'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen + 1 }, (_, i) => BigInt(i)), [1, seqlen + 1]);
                feed_backward['position_ids'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen + 1 }, (_, i) => BigInt(i)), [1, seqlen + 1]);
            }
        }
        if (this.profiler) {
            this.sess.endProfiling();
        }
        const output = [...this.output_tokens, ...[...this.output_tokens].reverse().slice(1)];
        console.log(output);
        return output;
    }
}

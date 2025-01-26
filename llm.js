import * as ort from 'onnxruntime-web/webgpu';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = document.location.pathname.replace('index.html', '');// + 'dist/';


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


class BeamState {
    constructor(tokens, feedForward, feedBackward, score = -1000, tri_block_dict = new Map()) {
        this.tokens = tokens;
        this.feedForward = feedForward;
        this.feedBackward = feedBackward;
        this.score = score;
        this.tri_block_dict = tri_block_dict;
    }

    clone() {
        const newFeedF = { ...this.feedForward };
        const newFeedB = { ...this.feedBackward };

        return new BeamState(
            [...this.tokens],
            newFeedF,
            newFeedB,
            this.score,
            new Map(Array.from(this.tri_block_dict.entries(), ([k, v]) => [k, [...v]]))
        );
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
        const provider_forward = "wasm";//options_forward.provider || "webgpu";
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

        const provider_backward = "wasm";//options_backward.provider || "webgpu";
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

    topKSamplingWithoutReplacement(
        logits_forward,
        logits_backward,
        block_ids,
        k,
        temperature,
    ) {
        const arrF = logits_forward.data;
        const arrB = logits_backward.data;

        const seq_len = logits_forward.dims[1];
        const vocab_size = logits_forward.dims[2];
        const start = vocab_size * (seq_len - 1);

        let scores = [];
        for (let i = 0; i < vocab_size; i++) {
            if (block_ids.includes(i)) {
                scores.push({ token: i, score: Number.NEGATIVE_INFINITY });
            } else {
                const val = Math.min(arrF[start + i], arrB[start + i]);
                if (!isFinite(val)) {
                    throw new Error("found infinite in logits");
                }
                scores.push({ token: i, score: val });
            }
        }

        scores.sort((a, b) => b.score - a.score);
        if (temperature < 0.001) {
            return scores.slice(0, k);
        }

        function computeExpVals(array) {
            const maxLogit = array[0].score / temperature;
            let sumExp = 0;
            for (let i = 0; i < array.length; i++) {
                const scaled = array[i].score / temperature;
                array[i].expVal = Math.exp(scaled - maxLogit);
                sumExp += array[i].expVal;
            }
            return sumExp;
        }

        let results = [];
        const sumExp = computeExpVals(scores);
        for (let round = 0; round < k; round++) {

            if (sumExp === 0 || scores.length === 0) {
                break;
            }

            let r = Math.random() * sumExp;
            let chosenIndex = 0;
            for (let i = 0; i < scores.length; i++) {
                if (r < scores[i].expVal) {
                    chosenIndex = i;
                    break;
                }
                r -= scores[i].expVal;
            }

            results.push({
                token: scores[chosenIndex].token,
                score: scores[chosenIndex].score
            });
            scores.splice(chosenIndex, 1);
        }

        return results;
    }

    async computePerplexityForward(tokens) {
        const input_ids = [2].concat(tokens);
        const seq_len = input_ids.length;

        let ppl_feed_forward = {};
        // モデル入力用のTensorに変換
        ppl_feed_forward['input_ids'] = new ort.Tensor(
            'int64',
            BigInt64Array.from(input_ids.map(BigInt)),
            [1, seq_len]
        );

        // position_ids, attention_maskなど必要に応じて設定
        if (this.need_position_ids) {
            ppl_feed_forward['position_ids'] = new ort.Tensor(
                'int64',
                BigInt64Array.from({ length: seq_len }, (_, i) => BigInt(i)),
                [1, seq_len]
            );
        }
        ppl_feed_forward['attention_mask'] = new ort.Tensor(
            'int64',
            BigInt64Array.from({ length: seq_len }, () => 1n),
            [1, seq_len]
        );

        // 1回だけ推論してすべてのlogitsを取得
        const outputs = await this.sess_forward.run(ppl_feed_forward);
        const logits = outputs.logits; // [1, seq_len, vocab_size]
        const arr = logits.data;
        const vocab_size = logits.dims[2];

        let totalNLL = 0.0;
        let count = 0;

        // ループは seq_len - 1 回 (直前トークン => 次トークンの予測確率)
        for (let t = 0; t < seq_len - 1; t++) {
            const start = t * vocab_size;
            const nextTokenId = input_ids[t + 1];

            // 数値安定のための log-sum-exp
            // 1) 行の最大値
            let maxLogit = -Infinity;
            const row = arr.subarray(start, start + vocab_size);
            for (const val of row) {
                if (val > maxLogit) maxLogit = val;
            }

            // 2) sumExp
            let sumExp = 0.0;
            for (const val of row) {
                sumExp += Math.exp(val - maxLogit);
            }

            const logSumExp = maxLogit + Math.log(sumExp);
            const logProbNext = row[nextTokenId] - logSumExp; // log-softmax

            totalNLL += -1 * logProbNext;
            count++;
        }

        const avgNLL = totalNLL / count;
        const ppl = Math.exp(avgNLL);
        return ppl;
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

    async generate(
        tokens,
        generate_num,
        beam_size,
        temperature,
        callback,
    ) {
        const max_tokens = generate_num;
        this.stop = false;

        const feed_forward_init = { ...this.feed_forward };
        const feed_backward_init = { ...this.feed_backward };

        const input_ids_forward = new ort.Tensor('int64', BigInt64Array.from(([2].concat(tokens)).map(BigInt)), [1, tokens.length + 1]);
        const input_ids_backward = new ort.Tensor('int64', BigInt64Array.from(([5].concat(tokens)).map(BigInt)), [1, tokens.length + 1]);
        feed_forward_init['input_ids'] = input_ids_forward;
        feed_backward_init['input_ids'] = input_ids_backward;

        let default_tri_block_dict = new Map()
        for (let i = 2; i < tokens.length; i++) {
            const key2 = `${tokens[i - 2]},${tokens[i - 1]}`;
            if (!default_tri_block_dict.has(key2)) {
                default_tri_block_dict.set(key2, [tokens[i]]);
            } else {
                default_tri_block_dict.get(key2).push(tokens[i]);
            }
        }

        const initTokens = BigInt64Array.from(([2].concat(tokens)).map(BigInt));
        const beam = new BeamState(initTokens, feed_forward_init, feed_backward_init, -1000, default_tri_block_dict);

        let beams = [beam];

        const input_len = input_ids_forward.size;
        const seqlen_init = input_len;
        if (this.need_position_ids) {
            beam.feedForward['position_ids'] = new ort.Tensor(
                'int64',
                BigInt64Array.from({ length: input_len }, (_, i) => BigInt(seqlen_init - input_len + i)),
                [1, input_len]
            );
            beam.feedBackward['position_ids'] = new ort.Tensor(
                'int64',
                BigInt64Array.from({ length: input_len }, (_, i) => BigInt(seqlen_init - input_len + i)),
                [1, input_len]
            );
        }
        let final_outputs = []
        while (!this.stop) {
            let newBeams = [];

            for (const b of beams) {
                const seqlen = b.tokens.length;

                if (seqlen >= max_tokens) {
                    continue;
                }

                b.feedForward['attention_mask'] = new ort.Tensor(
                    'int64',
                    BigInt64Array.from({ length: seqlen }, () => 1n),
                    [1, seqlen]
                );
                b.feedBackward['attention_mask'] = new ort.Tensor(
                    'int64',
                    BigInt64Array.from({ length: seqlen }, () => 1n),
                    [1, seqlen]
                );

                const outputs_forward = await this.sess_forward.run(b.feedForward);
                const outputs_backward = await this.sess_backward.run(b.feedBackward);

                let block_ids = [];
                if (seqlen >= 2) {
                    const key2 = `${b.tokens[seqlen - 2]},${b.tokens[seqlen - 1]}`;
                    if (b.tri_block_dict.has(key2)) {
                        block_ids.push(...b.tri_block_dict.get(key2));
                    }
                }

                const candidates = this.topKSamplingWithoutReplacement(outputs_forward.logits, outputs_backward.logits, block_ids, beam_size + 1, temperature);
                for (let c of candidates) {
                    const newBeam = b.clone();
                    newBeam.tokens.push(BigInt(c.token));
                    newBeam.score = newBeam.score + c.score;
                    if (seqlen >= 2) {
                        const key2 = `${b.tokens[seqlen - 2]},${b.tokens[seqlen - 1]}`;
                        if (!newBeam.tri_block_dict.has(key2)) {
                            newBeam.tri_block_dict.set(key2, [c.token]);
                        } else {
                            newBeam.tri_block_dict.get(key2).push(c.token);
                        }
                    }

                    this.update_kv_cache(newBeam.feedForward, outputs_forward);
                    this.update_kv_cache(newBeam.feedBackward, outputs_backward);

                    const newSeqLen = newBeam.tokens.length;
                    newBeam.feedForward['input_ids'] = new ort.Tensor(
                        'int64',
                        BigInt64Array.from(newBeam.tokens),
                        [1, newSeqLen]
                    );
                    newBeam.feedBackward['input_ids'] = new ort.Tensor(
                        'int64',
                        BigInt64Array.from([BigInt(5), ...newBeam.tokens.slice(1)]),
                        [1, newSeqLen]
                    );

                    if (this.need_position_ids) {
                        newBeam.feedForward['position_ids'] = new ort.Tensor(
                            'int64',
                            BigInt64Array.from({ length: newSeqLen }, (_, i) => BigInt(i)),
                            [1, newSeqLen]
                        );
                        newBeam.feedBackward['position_ids'] = new ort.Tensor(
                            'int64',
                            BigInt64Array.from({ length: newSeqLen }, (_, i) => BigInt(i)),
                            [1, newSeqLen]
                        );
                    }

                    newBeams.push(newBeam);
                }

            }
            if (newBeams.length === 0) {
                break;
            }

            newBeams.sort((a, b) => b.score - a.score);
            final_outputs.push({ beam: newBeams[0], score: Number(await this.computePerplexityForward([...newBeams[0].tokens, ...[...newBeams[0].tokens].reverse().slice(1)])) });
            beams = newBeams.slice(0, beam_size);

            if (callback && !this.profiler) {
                const bestBeam = beams[0];
                const bestTokens = Array.from(bestBeam.tokens);
                const temp_output = [...bestTokens, ...bestTokens.reverse().slice(1)];
                callback(temp_output);
            }

            if (beams.every(b => b.tokens.length >= max_tokens)) {
                break;
            }
        }

        //beams.sort((a, b) => b.score - a.score);
        //const bestBeam = beams[0];
        final_outputs.sort((a, b) => -b.score + a.score);
        const bestBeam = final_outputs[0].beam;
        const output = [...bestBeam.tokens, ...[...bestBeam.tokens].reverse().slice(1)];

        return output;
    }
}

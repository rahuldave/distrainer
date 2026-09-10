# An introduction to distributed training with Ray, and how distrainer does it

This is a plain-language tour. It starts from zero, introduces each term the first time it is used, and ends with how distrainer's design differs from what Ray and Anyscale give you out of the box. If you already know what DDP and an all-reduce are, skip to [Part 4](#part-4-ray-train-and-pytorch).

## Part 1: Why train on more than one machine

Training a neural network means repeating one loop millions of times: take a small batch of examples, run them through the model, measure how wrong the model was, nudge the model's weights so it is a little less wrong, repeat. A **batch** is the handful of examples processed in one nudge; a **step** is one trip through the loop; an **epoch** is one full pass over all of the training data.

A single GPU can only hold so much and only go so fast. If you want to train faster, or on more data, you use several GPUs, often spread across several computers. Each computer is called a **node**. Each running copy of your training program is a **worker**. With four nodes of one GPU each you have four workers. The set of all workers is the **world**, and each worker has a number from `0` to `world_size - 1` called its **rank**. Rank 0 usually gets a few extra jobs, such as saving files, so you do not save four copies.

The moment there is more than one worker you have to answer two questions:

1. **Who works on which data?** If all four workers process the same batch, you have done nothing useful.
2. **How do the workers agree on the model?** Each worker nudges its own copy of the weights. If they never talk, you end up with four different models.

Everything in distributed training is an answer to those two questions.

## Part 2: The vocabulary of parallel programs

Before looking at training specifically, here are the handful of communication patterns every parallel program is built from. The names come from the message-passing world (MPI) and PyTorch uses the same ones.

**Map** — apply the same function to many pieces of data independently. No communication needed. "Resize every image in this folder" is a map.

**Reduce** — combine many values into one with an operation like sum or max. "Add up the word counts from every file" is a reduce. Map followed by reduce is the MapReduce pattern that big-data systems are built on.

```mermaid
flowchart LR
  subgraph map["map: same function, different inputs"]
    a1[x1] --> f1[f] --> y1[y1]
    a2[x2] --> f2[f] --> y2[y2]
    a3[x3] --> f3[f] --> y3[y3]
  end
  subgraph reduce["reduce: combine into one"]
    y1 --> s((sum))
    y2 --> s
    y3 --> s
    s --> total[y1+y2+y3]
  end
```

**Reading the diagram.** On the left, three inputs go through the same function `f` at the same time and produce three outputs; no worker needs to know about the others. On the right, the three outputs are combined by one operation, a sum, into a single result. Distributed training uses both shapes: forward and backward passes on different batches are a map, and averaging the gradients is a reduce.

The next four are **collectives**: operations that every worker in the world participates in at the same time. A collective is also a meeting point; no worker can finish it until all of them have arrived. This is what people mean by a **barrier**.

**Broadcast** — one worker sends the same value to everyone. Used to copy the initial model weights from rank 0 to all ranks so everyone starts identical.

**Scatter** — one worker splits a list and sends piece *i* to rank *i*. Everyone gets a different slice.

**Gather** — the reverse of scatter: every worker sends its piece to one worker, which assembles the list.

**All-gather** — like gather, but *every* worker ends up with the full assembled list, not just one of them.

**All-reduce** — every worker contributes a value, they are reduced (say, averaged), and *every* worker receives the result. This is the single most important collective in training.

```mermaid
flowchart TB
  subgraph broadcast["broadcast"]
    b0["rank 0: W"] --> b1["rank 1: W"]
    b0 --> b2["rank 2: W"]
    b0 --> b3["rank 3: W"]
  end
  subgraph scatter["scatter"]
    s0["rank 0: [a,b,c,d]"] --> s1["rank 1: b"]
    s0 --> s2["rank 2: c"]
    s0 --> s3["rank 3: d"]
    s0 --> s00["rank 0: a"]
  end
```

**Reading the diagram.** In the broadcast, rank 0 holds the value `W` and afterwards every rank holds the same `W`; nothing is split. In the scatter, rank 0 holds a list of four items and afterwards each rank holds exactly one of them, including rank 0 keeping the first.

```mermaid
flowchart TB
  subgraph allgather["all-gather: everyone ends up with everything"]
    g0["rank 0: a"] --> r0["rank 0: [a,b,c,d]"]
    g1["rank 1: b"] --> r0
    g2["rank 2: c"] --> r0
    g3["rank 3: d"] --> r0
    g0 --> r1["rank 1: [a,b,c,d]"]
    g1 --> r1
    g2 --> r1
    g3 --> r1
  end
  subgraph allreduce["all-reduce (average): everyone ends up with the same mean"]
    v0["rank 0: g0"] --> m["mean(g0,g1,g2,g3)"]
    v1["rank 1: g1"] --> m
    v2["rank 2: g2"] --> m
    v3["rank 3: g3"] --> m
    m --> o0["rank 0"]
    m --> o1["rank 1"]
    m --> o2["rank 2"]
    m --> o3["rank 3"]
  end
```

**Reading the diagram.** In the all-gather each rank starts with one item and every rank ends with the whole list, so it is a gather whose result is delivered to everyone. In the all-reduce each rank starts with its own gradient, the four gradients are averaged, and every rank receives that one mean. The all-reduce is drawn with a single combining node because that is what happens logically; in practice the ranks exchange pieces directly with each other and no single rank holds all four gradients.

Two practical facts about collectives matter for what follows. First, they are only fast on GPUs because of special libraries (NCCL on NVIDIA hardware) that move data directly between GPUs. Second, because a collective is a barrier, every rank must call it the same number of times and in the same order, or the program hangs forever. That rule will show up again when we talk about checkpoints.

## Part 3: Data-parallel training (DDP)

The most common way to use many workers is **data parallelism**: every worker holds a complete copy of the model, each worker gets a *different* batch, and after computing its gradients (the "which way to nudge" vector) the workers **all-reduce** the gradients so everyone applies the same averaged update. Because everyone started identical and applies identical updates, the copies stay identical. PyTorch's implementation is called **DistributedDataParallel**, or **DDP**.

```mermaid
sequenceDiagram
  participant R0 as rank 0
  participant R1 as rank 1
  participant R2 as rank 2
  Note over R0,R2: all hold identical model weights W
  R0->>R0: forward+backward on batch A → gradient gA
  R1->>R1: forward+backward on batch B → gradient gB
  R2->>R2: forward+backward on batch C → gradient gC
  R0->>R2: all-reduce, g = mean(gA, gB, gC)
  Note over R0,R2: every rank applies W ← W − lr·g
  Note over R0,R2: weights still identical, so the next step starts in sync
```

**Reading the diagram.** Three ranks start with identical weights. Each computes a gradient from its own batch; the horizontal arrow is the all-reduce that replaces all three gradients with their mean; then each rank applies the identical update. The final note is the invariant that makes data parallelism work: because the update was identical, the weights are still identical, and the next step can begin without any further synchronization.

The two questions from Part 1 are answered like this: *who works on which data* is solved by giving each rank a different slice, its **shard**, and *how the workers agree* is solved by the all-reduce. The effective batch size is the per-worker batch times the world size, which is why learning rates are usually scaled when you add workers.

Three other patterns exist and are worth knowing by name, though distrainer v0.1 focuses on data parallelism. **Model parallelism** splits one model's layers across GPUs when it does not fit on one. **Fully Sharded Data Parallel (FSDP)** keeps data parallelism but *shards* the weights and optimizer state across workers to save memory, gathering each layer's weights just in time with an all-gather. **Pipeline parallelism** turns the layers into an assembly line. All of them still need a per-rank data shard, so everything said below about data applies to them too.

## Part 4: Ray Train and PyTorch

**Ray** is a system for running Python across a cluster. You start a **head node** and any number of worker nodes, and Ray lets you launch functions (**tasks**) and long-lived objects (**actors**) anywhere in the cluster while it handles scheduling and moving data between nodes. **Ray Train** is the library on top of Ray for distributed training; **Ray Data** is the library for loading and transforming data. Anyscale is the company behind Ray and sells a hosted version with extras (Part 7).

Ray Train's shape is simple. You write one Python function, the **training function**, that contains your normal PyTorch loop. Ray Train starts a **controller** process on the head node, the controller launches one worker actor per rank (each with one GPU), and every worker runs your training function. Ray Train sets up the process group so PyTorch's collectives work, and gives you a few helpers inside the function.

```mermaid
flowchart TB
  driver["your script: TorchTrainer(train_func, ScalingConfig(num_workers=3)).fit()"]
  driver --> ctrl["Train controller (head node)<br/>starts workers, watches for failures,<br/>tracks checkpoints"]
  ctrl --> w0["worker rank 0<br/>runs train_func"]
  ctrl --> w1["worker rank 1<br/>runs train_func"]
  ctrl --> w2["worker rank 2<br/>runs train_func"]
  w0 <-. NCCL all-reduce .-> w1
  w1 <-. NCCL all-reduce .-> w2
```

**Reading the diagram.** Your script runs on the head node and asks for three workers. The controller, also on the head node, creates the three worker processes, gives each a rank, sets up the process group, and then watches them; it does no training itself. Each worker runs your training function. The dotted lines between workers are the NCCL collectives (the all-reduce from Part 3), which go directly between GPUs without passing through the controller.

The helpers you use inside the training function:

- `ray.train.torch.prepare_model(model)` wraps your model in DDP and moves it to the right GPU.
- `ray.train.get_dataset_shard("train")` hands this rank its slice of the data (Part 5). If you prefer a plain PyTorch `DataLoader`, `prepare_data_loader` adds the sampler that makes each rank see a different slice.
- `ray.train.report(metrics, checkpoint=...)` sends metrics to the controller and, optionally, saves a **checkpoint**: a folder with the model weights and whatever else you need to continue later. `report` is a barrier, so every rank must call it the same number of times.
- `ray.train.get_checkpoint()` returns the latest checkpoint when the run is restarting after a failure, so you can load it and continue.

Outside the function, `ScalingConfig` says how many workers and whether they use GPUs, `RunConfig` says where checkpoints go (a local folder, S3, or any S3-compatible store), and `FailureConfig` says how many times to retry after a failure. Retrying means: throw away all workers, start new ones, call the training function again, and let it load the latest checkpoint.

## Part 5: How data loading works in Ray

Ray Data represents a dataset as a list of **blocks**, each a table of a few thousand rows stored in Ray's shared memory. Reading files, filtering, and transforming are maps over blocks, run as Ray tasks across the cluster, streamed rather than loaded all at once.

When you hand a dataset to Ray Train, it is **split** across workers: rank *i* gets `1/world_size` of it. The important thing to understand is *how*. The split is dynamic. A single coordinator runs the dataset pipeline and hands each finished block to whichever worker asks next, preferring a worker on the same node as the block so less data crosses the network. So a worker's shard is not "rows 0 to 25 percent"; it is "whichever blocks this worker happened to pull this time". The coordinator also equalizes the row counts so every rank takes the same number of steps, dropping a few leftover rows if needed.

```mermaid
flowchart LR
  files[(parquet files)] --> pipe["Ray Data pipeline<br/>read → map → shuffle"]
  pipe --> coord["split coordinator<br/>hands out blocks on request"]
  coord -- block 7 --> w0["rank 0 shard"]
  coord -- block 3 --> w1["rank 1 shard"]
  coord -- block 11 --> w2["rank 2 shard"]
  w0 --> b0["iter_torch_batches(batch_size=B)"]
  w1 --> b1["iter_torch_batches(batch_size=B)"]
  w2 --> b2["iter_torch_batches(batch_size=B)"]
```

**Reading the diagram.** Data flows left to right. The pipeline reads files and applies transformations block by block. The split coordinator is a single process that receives finished blocks and hands each to whichever rank asks next, so the block numbers next to the arrows are arbitrary: rank 0 happened to get block 7, rank 1 block 3. Each rank's shard is therefore the sequence of blocks it happened to receive, and `iter_torch_batches` slices that sequence into fixed-size batches on the fly.

Inside the training function, `shard.iter_torch_batches(batch_size=B)` walks the incoming blocks and slices them into batches of `B` rows, prefetching in the background. When your loop finishes the shard and starts again, that is a new epoch: all ranks wait at a barrier, and the whole pipeline re-executes from the files. This is why expensive preprocessing should be done once and stored, with only cheap per-epoch work such as shuffling left in the pipeline.

Note what is *not* here: there is no notion of "which batch am I on" that survives a restart. If a worker dies at step 3,000 of a 10,000-step epoch and Ray restarts the run from a checkpoint, the pipeline re-executes and the shard starts from the beginning of the epoch. The model weights come back from the checkpoint; the data position does not.

## Part 6: Checkpoints, failures, and elasticity

A checkpoint is the run's save game. Ray Train writes it wherever `RunConfig` points, keeps the last few, and, when any worker fails, restarts all workers and hands the newest checkpoint back. Node failures and **preemption** (a cloud provider taking back a cheap "spot" machine) are handled the same way.

Ray Train can also be **elastic**: you say "between 2 and 8 workers", the run starts as soon as 2 are available, and when more machines join or some leave, the controller restarts the workers at the new size from the latest checkpoint. That is powerful but it makes the data-position problem worse: if you had 4 workers and now have 3, a saved "rank 2 was at step 3,000" no longer means anything, because the data is re-split three ways.

So the honest summary of open-source Ray is: excellent at restarting the *model*, silent about restarting the *data*. You either accept that a restart repeats part of an epoch, or you build something.

## Part 7: What Anyscale adds

Anyscale's hosted runtime adds features on top of open-source Ray. For training, the relevant ones are:

- **Mid-epoch resumption.** Every row gets a unique id column, the runtime records which ids each worker has consumed, and on restart the dataset skips them. The training function saves `shard.state_dict()` next to the model and passes it back on restart. It only works for pipelines made of simple per-row transformations (no shuffles or joins in the middle), needs shared storage for its bookkeeping, and is a beta feature.
- **A training dashboard** with per-worker logs, progress, error attribution, and profiling; logs kept for 30 days.
- **Cluster-level handling of spot preemption**, job queues, priority scheduling, and alerting.

Elastic training used to be on this list; it is now in open-source Ray Train as well. The one that matters for the "where was I in the data" problem is mid-epoch resumption, and that is the gap distrainer fills, in a different way.

## Part 8: How distrainer works

distrainer keeps Ray Train for everything it is good at (launching workers, process groups, checkpoints, retries, elasticity, S3 storage) and replaces one thing: how data reaches the workers. The idea is to make a **block** the unit of everything, so that progress is simply "how many blocks are done", a number that does not care how many workers there are.

### The block

In distrainer a block is a pre-built batch of rows with a stable name, its `block_id`, stored as one Parquet file in a **block store** (a folder or an S3 bucket). Blocks are built ahead of time by whatever process you like, typically a Ray Data job. The training loop never sees loose rows; it sees blocks. A block can be exactly one GPU batch, or a larger "super-batch" that the worker cuts into several micro-batches. Which rows belong to which block is not stored anywhere special: a row belongs to the block whose file it sits in, and each row also carries its `block_id` as a column so you can always tell where it came from.

### The block log: batch and streaming are the same thing

Somebody has to decide the *order* in which blocks are trained on. distrainer stores that order as a **log**: an append-only list of blocks, written in groups called **segments**. A segment is a fixed number of blocks, `W`, say 256. When a segment is written, its `W` blocks are shuffled amongst each other using a seed, and the shuffled order is saved in a small file, `log/00000007.json` for segment 7. Once written, a segment never changes.

That one structure covers both ways data can arrive. In **batch mode**, all of the data exists before training starts, so the whole log is written up front, every segment at once, and an `_END` marker says "that is everything". In **streaming mode**, blocks are being produced while training runs (by a miner that keeps finding new hard negatives, say), so segments are appended as they fill up and there is no `_END` until the producer stops. The trainer does not care which mode it is in. It reads segment 0, then segment 1, and when it reaches a segment that does not exist yet it waits for it to appear. That waiting is the only difference between the two modes, and it is invisible to the training code.

There is no database and no message broker behind this; a segment is a file, and writing a file is atomic on both a local disk and S3, so a reader either sees a complete segment or none. The only rule is that exactly one process writes to a given log, so the order is never ambiguous.

```mermaid
flowchart LR
  subgraph store["block store (folder or S3 bucket)"]
    direction TB
    blocks["blocks/<br/>b0001.parquet b0002.parquet … (one file = one batch)"]
    log["log/<br/>00000000.json  00000001.json  00000002.json  _END?"]
  end
  producer["producer / writer<br/>(batch: writes everything at start;<br/>streaming: keeps appending)"] -->|"W blocks, shuffled, then commit"| log
  producer --> blocks
  log -->|"read segment k, wait for k+1"| trainer["trainer ranks"]
  blocks -->|"each rank fetches its own blocks"| trainer
```

**Reading the diagram.** Everything lives in one folder or bucket. The producer writes block files into `blocks/` and, once it has `W` of them, commits one segment file into `log/` listing those blocks in shuffled order; in batch mode it does this for the whole dataset before training starts and adds `_END`, in streaming mode it keeps going. The trainer ranks read the segment files in numerical order, fetch the blocks each segment names, and, if the next segment file does not exist yet, simply wait for it. The producer and the trainer never talk to each other directly; the files are the only channel.

### Dealing a segment to the ranks

Inside a segment, blocks are dealt to ranks like cards: position `i` goes to rank `i mod world_size`, at step `i div world_size`. Each rank's cards are its **lane**, and it fetches them itself from the block store with a small prefetching loader. There is no coordinator and no lockstep beyond the all-reduce that DDP already does. Because `W` is chosen as a multiple of the number of workers, every rank gets exactly `W / world_size` blocks per segment and nothing is left over. The example below uses `W = 12` and three workers, so a segment is four steps long; the same numbers are used in every diagram that follows.

```mermaid
flowchart LR
  seg["segment 2 (W = 12, shuffled):<br/>b17 b03 b42 b08 b25 b31 b11 b06 b19 b27 b02 b34"]
  seg --> r0["rank 0 lane: b17 b08 b11 b27"]
  seg --> r1["rank 1 lane: b03 b25 b06 b02"]
  seg --> r2["rank 2 lane: b42 b31 b19 b34"]
  r0 --> s["step 0: b17 b03 b42 → all-reduce → update<br/>step 1: b08 b25 b31 → all-reduce → update<br/>step 2: b11 b06 b19 → all-reduce → update<br/>step 3: b27 b02 b34 → all-reduce → update"]
  r1 --> s
  r2 --> s
```

**Reading the diagram.** The segment's twelve blocks are listed in the order the writer committed them. Dealing is round-robin: the first block goes to rank 0, the second to rank 1, the third to rank 2, the fourth back to rank 0, and so on, which gives each rank a lane of four blocks. The right-hand box regroups the same blocks by step: at step 0 the three ranks train on `b17`, `b03`, `b42` simultaneously, average their gradients, and update; then step 1, and so on. Reading down a lane gives one rank's view; reading across a step gives the whole world's view.

So the units nest like this: a **row** is one example; a **block** is one rank's batch; a **step** is `world_size` blocks, one per rank, ending in the gradient all-reduce and one weight update; a **segment** is `W` blocks shuffled together, and the place where the "ingredients" can change (new hard negatives, a new seed); and the **log** is the whole sequence of segments. Gradients are averaged at every step. Nothing about the model is communicated at segment boundaries; those are purely about data.

Here is the whole hierarchy with the example numbers. Positions count from the start of the log, so segment 2 covers positions 24 to 35; each step takes the next three positions, one per rank; and a position is one block, which is one Parquet file of, say, 32 rows.

```mermaid
flowchart LR
  s0["segment 0<br/>positions 0–11"] --> s1["segment 1<br/>positions 12–23"] --> s2["segment 2<br/>positions 24–35"] --> s3["segment 3 …<br/>(batch mode ends with _END)"]
  s2 ==> seg
  subgraph seg["segment 2 = W = 12 blocks, shuffled together when written (log/00000002.json)"]
    direction TB
    st0["step 0 · positions 24 25 26<br/>rank 0: b17 · rank 1: b03 · rank 2: b42<br/>→ all-reduce → update"]
    st1["step 1 · positions 27 28 29<br/>rank 0: b08 · rank 1: b25 · rank 2: b31<br/>→ all-reduce → update"]
    st2["step 2 · positions 30 31 32<br/>rank 0: b11 · rank 1: b06 · rank 2: b19<br/>→ all-reduce → update"]
    st3["step 3 · positions 33 34 35<br/>rank 0: b27 · rank 1: b02 · rank 2: b34<br/>→ all-reduce → update"]
    st0 --> st1 --> st2 --> st3
  end
  seg -.-> blk["one block, e.g. b08 = blocks/b08.parquet<br/>= rank 0's batch at step 1<br/>= 32 rows, each carrying block_id = b08"]
```

**Reading the diagram.** The bottom row is the log: segments 0, 1, 2, … in the order the writer committed them, each owning twelve consecutive positions. Segment 2 is opened up in the box: its twelve blocks are already in their shuffled order, and they are consumed three at a time, one per rank, as steps 0 to 3. Each step ends with the gradient all-reduce and one weight update, so there are exactly four updates per segment with three workers. The dotted arrow picks out one block, `b08`: it is position 27, it is rank 0's batch at step 1, and on disk it is a single Parquet file of 32 rows that all carry `block_id = b08`. Nothing about the model is communicated at the segment boundary; that boundary is only where the data ingredients may change and where a hook may run.

### The ledger

Every checkpoint carries a tiny file called the **ledger**: which `segment` we are in, the `cursor` (how many steps of that segment are done), and the `world_size` at the time. Because `report` is a barrier, all ranks are at the same step whenever a checkpoint is written, so this one small number describes the data position for the whole world. Resuming means opening the same segment file, skipping the first `cursor × world_size` positions, and re-dealing the rest, and every later segment, over however many workers there are *now*. Continuing the example: a checkpoint taken after step 1 of segment 2 says `{segment 2, cursor 2, world_size 3}`, meaning positions 24 to 29 are done.

```mermaid
flowchart TB
  ck["last checkpoint: weights + ledger {segment 2, cursor 2, world_size 3}<br/>= positions 24–29 (b17 b03 b42 b08 b25 b31) are done"]
  ck --> f["worker dies during step 3, or the run is resized from 3 to 2 workers"]
  f --> reopen["reopen log/00000002.json (immutable, so the same 12 blocks in the same order)"]
  reopen --> skip["skip 2 × 3 = 6 positions (24–29)"]
  skip --> deal["deal positions 30–35 over 2 ranks:<br/>step 0: b11 b06 · step 1: b19 b27 · step 2: b02 b34<br/>then segment 3, 4, …"]
  deal --> replay["only b11 b06 b19 (step 2 of the old run) is trained twice;<br/>with EveryKSteps(1) nothing would be"]
```

**Reading the diagram.** The run last checkpointed after step 1 of segment 2, so the ledger says `cursor 2` with `world_size 3`: six positions, 24 to 29, are done. Then something interrupts the run during step 3 (a worker dies, or the world is resized from three workers to two). On restart the trainer reopens the same segment file, which cannot have changed, skips the six finished positions, and deals the remaining six over the two ranks that exist now: three steps instead of two. The blocks of old step 2 (`b11 b06 b19`) had been trained on but not yet checkpointed, so they are trained a second time; that replay is the whole cost of the failure, and a tighter checkpoint policy makes it smaller.

The only data replayed is whatever ran between the last checkpoint and the failure, which you control with the checkpoint cadence. That is the same guarantee Anyscale's mid-epoch resumption gives, without tracking per-row ids, and it survives a change in world size, which per-rank iterator state does not. In streaming mode this adds one requirement: a segment must not be deleted until a later checkpoint exists, so that a replay can still find its blocks. distrainer's garbage collector keeps a configurable number of segments behind the latest checkpoint.

### The checkpoint policy

Because progress is counted in steps, "when to checkpoint" becomes a plain rule: every K steps, at the end of each segment, at the end of each pass over the data, or a time budget. Rules based on indices need no communication at all; every rank computes the same answer. A time-based rule is decided by rank 0 and broadcast so the ranks agree. Uploads to S3 run in a background thread so frequent checkpoints do not stall training.

With the example numbers, here is where each policy would save during segment 2. Every step boundary is a legal checkpoint because every rank has just finished the same step; the policy only decides which of those boundaries to use. Each ledger value shown is what a checkpoint written at that point would contain.

```mermaid
flowchart LR
  subgraph seg["segment 2, world_size = 3 (4 steps of 3 blocks)"]
    direction LR
    st0["step 0<br/>b17 b03 b42"] --> c0{{"ledger<br/>{seg 2, cursor 1, ws 3}"}}
    c0 --> st1["step 1<br/>b08 b25 b31"] --> c1{{"ledger<br/>{seg 2, cursor 2, ws 3}"}}
    c1 --> st2["step 2<br/>b11 b06 b19"] --> c2{{"ledger<br/>{seg 2, cursor 3, ws 3}"}}
    c2 --> st3["step 3<br/>b27 b02 b34"] --> c3{{"ledger<br/>{seg 2, cursor 4, ws 3}"}}
  end
  c3 --> hook["segment end: hook runs (re-mine, append segment 3),<br/>then everyone reads segment 3"]
  ek["EveryKSteps(2) saves here"] -.-> c1
  ek -.-> c3
  se["SegmentEnd saves here"] -.-> c3
  ev["EveryKSteps(1) saves at every one"] -.-> c0
```

**Reading the diagram.** Time runs left to right through segment 2. After each step every rank has finished the same block, so each of the four hexagons is a legal place to write a checkpoint, and the ledger value shown is exactly what that checkpoint would record: the segment, how many steps of it are done, and the world size. The dashed arrows show which of those boundaries three different policies would choose: `EveryKSteps(1)` saves at every hexagon, `EveryKSteps(2)` at the second and fourth, `SegmentEnd` only at the fourth. With `Any([EveryKSteps(2), SegmentEnd])` you get the union. After the last step the segment hook runs on rank 0 (for example re-mining and appending segment 3) while the other ranks wait, and then everyone moves on to segment 3.

### Segment hooks

The end of a segment is a natural place to do sub-epoch work. distrainer lets rank 0 run a hook there, for example: embed the whole corpus with the current model, find new hard examples, and *append the next segment to the log*. In other words, a hook is just a producer that happens to live inside the training job. The other ranks wait at a barrier until the segment is committed, then everyone reads it. The same hook could equally run as a separate process writing to the same log; the trainer cannot tell the difference.

## Part 9: Why this is a better fit than Anyscale's approach (for designed-batch training)

Anyscale solves the resume problem by recording every row id that was consumed and filtering them out on restart. distrainer solves it by making the data assignment deterministic in the first place, so the position is a single number. The consequences:

- **World-size independence.** A change from 4 to 3 workers is a re-deal, not a special case. Anyscale's row-id approach also handles this, but per-rank iterator state in general does not.
- **Nothing to track at row level.** No id column requirement, no sidecar bookkeeping on shared storage, no restriction to map-only pipelines. The pipeline can do anything it likes, because it runs *before* blocks are written.
- **Batch composition is yours.** A block is a batch you designed. Contrastive training with hard negatives, curriculum ordering, mixing several datasets in a chosen ratio, all become "what the writer puts in a segment", not "what the shuffle happened to produce".
- **Shuffling is at block level and reproducible.** Two runs with the same seed consume identical block sequences per rank.
- **It runs on open-source Ray**, on a laptop, in Docker, or on Kubernetes, and stores blocks and checkpoints on any S3-compatible service.

The trade-off is that data must be pre-blocked. If your training set is a live streaming pipeline with a global shuffle in the middle, Ray Data's streaming split is the better tool and Anyscale's row tracking is the way to resume it. For workloads where you *want* to control what is in a batch, pre-blocking is not a cost; it is the point.

## Part 10: What the block abstraction makes possible

Because every rank consumes "one block per global step" and nothing else is assumed about the block's contents, the same machinery supports many training styles by changing only what a block holds and what `train_step` does with it.

**Plain DDP.** A block is a batch; `train_step` does forward, backward, and lets DDP all-reduce the gradients. This is the default.

**Gradient accumulation.** A block holds several micro-batches; `train_step` loops over them, accumulating gradients, and steps the optimizer once. Checkpoint accounting stays per block.

**Contrastive learning with hard negatives.** A block holds anchors, positives, and the hard negatives mined for them. In-block negatives cost no communication. If you also want negatives from other ranks, `train_step` does an all-gather of embeddings across the world, and the writer can place related blocks at adjacent positions in a segment so they land in the same global step.

**Curriculum learning.** The writer orders segments from easy to hard instead of shuffling uniformly across the whole corpus.

**Multi-dataset mixing.** The writer interleaves blocks from several sources in a chosen ratio when it fills a segment; the ledger does not care where a block came from.

**Streaming data.** Blocks produced while training runs are appended to the same log; the trainer waits for the next segment instead of stopping, and resume works exactly as in batch mode.

**FSDP and other sharded-model schemes.** The data side is unchanged: one block per rank per step. Each rank reports its own checkpoint shard and Ray Train merges them into one checkpoint folder, and the ledger rides along.

**Evaluation blocks.** A block flagged as evaluation is run without gradients; the same lane loader serves it.

**Elastic and spot-instance training.** Any of the above, with `num_workers=(min, max)`; a resize is a re-deal.

## Glossary

- **Batch / step / epoch** — examples per update; one update; one pass over the data.
- **Node, worker, rank, world size** — a machine; a running copy of the program; its number; how many there are.
- **Map, reduce** — apply a function to many items; combine many values into one.
- **Broadcast, scatter, gather, all-gather, all-reduce** — the collective communication patterns (Part 2).
- **Barrier** — a point every rank must reach before any can continue; every collective is one.
- **DDP** — data-parallel training with full model copies and gradient all-reduce.
- **FSDP** — data parallelism with model weights sharded across ranks.
- **Ray, Ray Train, Ray Data** — cluster runtime; its training library; its data library.
- **Controller** — Ray Train's process on the head node that launches workers and tracks checkpoints.
- **Shard** — a rank's slice of the data.
- **Checkpoint** — a saved folder of weights and state used to resume.
- **Elastic training** — a run whose number of workers can change while it runs.
- **Preemption** — a cloud provider reclaiming a spot machine.
- **Block (Ray Data)** — a few thousand rows in shared memory, the unit Ray Data streams.
- **Block (distrainer)** — a pre-built, named batch stored as a Parquet file; a row belongs to the block whose file it is in.
- **Log** — the ordered, append-only sequence of segments that says in what order blocks are trained on; the same structure for batch and streaming data.
- **Segment** — `W` blocks shuffled amongst each other and committed as one immutable file; the sub-epoch unit where hooks and re-shuffles happen (also called a chunk).
- **Writer / producer** — the single process that appends segments to a log; in batch mode it writes everything at the start.
- **Lane** — one rank's share of a segment, dealt round-robin by position.
- **Ledger, cursor** — the data position stored in a checkpoint: which segment, and how many steps of it are complete.
- **Checkpoint policy** — the rule that decides at which step boundaries to save.
- **Segment hook** — code that rank 0 runs at the end of a segment, e.g. re-mining hard negatives and appending the next segment.
- **Retention** — how many old segments are kept so a replay after failure can still find its blocks.

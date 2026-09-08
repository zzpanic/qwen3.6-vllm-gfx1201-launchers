# KV-Cache Operations Runbook

> **What this is.** The three things you asked to be able to do yourself, written so
> you can do them without me: **turn the disk (L3) tier off**, **raise `/dev/shm` to
> 32 GiB**, and **set the KV tier to 30 GiB**. Each one is a numbered procedure with
> the exact command, what to expect, how to check it worked, and how to undo it.
>
> Companion docs: `kv-cache-handover.md` (the map), `kv-cache-known-issues.md`
> (what is broken), `kv-cache-current-implementation.md` (how it is wired).
>
> Last verified against the live box: **2026-09-09**.

---

## 0. The state right now (so you can tell if something moved)

| Thing | Value | Where it is set |
|---|---|---|
| Guest `MemTotal` | **39.17 GiB** | TrueNAS VM allocation (40 GiB configured) |
| `/dev/shm` limit | **28.00 GiB** | `/etc/fstab` line 23 |
| KV tier requested | **24 GiB** | `<your llama-swap config.yaml>`, `EXTRA=--kv-offloading-size 24` |
| KV tier **actually allocated** | **23.99 GiB** | `/dev/shm/vllm_offload_*.mmap` |
| RAM left over | **~7 GiB available** | `free -g` |
| L3 disk tier | **ON**, `/kvcache`, 28 GiB used of 511 GiB | `KVOFF_DISK=/kvcache` in config.yaml |
| fs tier serving | **ON** | `KVOFF_PENDING_IS_MISS=0` |
| CPU tier policy | **ARC** | `KVOFF_POLICY=arc` |

One-liner to re-read all of that at any time:

```bash
awk '/MemTotal/{printf "MemTotal      : %.2f GiB\n",$2/1048576}' /proc/meminfo
python3 -c "import os;s=os.statvfs('/dev/shm');print('/dev/shm limit: %.2f GiB'%(s.f_blocks*s.f_frsize/2**30))"
ls -la /dev/shm/vllm_offload_*.mmap | awk '{printf "tier region   : %.2f GiB\n",$5/2**30}'
free -g | awk '/^Mem:/{print "RAM available : "$7" GiB"}'
df -h /kvcache | tail -1 | awk '{print "L3 disk       : "$3" used of "$2" ("$5")"}'
```

---

## 1. Turn the disk (L3) tier OFF completely

**What it does.** Removes the third tier. The GPU cache (L1) and the 24 GiB RAM tier
(L2) keep working exactly as they do now — this is *not* "turn caching off", it is
"stop spilling to disk". You lose the ~484 GiB of overflow capacity and keep all the
speed. This is the revert I told you about when we enabled fs-tier serving: if the
extra ~4 s on a disk-served re-read is not worth it, this is the switch.

**Why you might.** The disk reads at ~117 MB/s against a ~101 MB/s break-even, i.e.
only **1.16×** faster than just recomputing the tokens. It is a real win but a thin
one, and it is the only thing that makes a re-read *slower* than it was before.
Turning it off costs you nothing but capacity.

### Do it

Edit `<your llama-swap config.yaml>`. Find the line (it is inside the `"qwen3.8-27b-vllm"` entry):

```yaml
      - "KVOFF_DISK=/kvcache"
```

Comment it out by putting a `#` at the front of the line, keeping the indentation:

```yaml
      #- "KVOFF_DISK=/kvcache"
```

Then restart:

```bash
sudo -n systemctl restart llama-swap
```

**Then check for an orphan — this bites every time:**

```bash
podman ps --format '{{.Names}}\t{{.Status}}'
```

If you see a `qwen38-27b-vllm` that predates the restart, it is an orphan holding the
GPU, the port and the `/dev/shm` region. Kill it:

```bash
podman stop -t 30 qwen38-27b-vllm
```

### Check it worked

```bash
sudo -n journalctl -u llama-swap --since "10 min ago" --no-pager | grep -i "kv-offload"
```

You want to **see** the CPU tier line and **not see** any fs-tier line:

- present: `[kv-offload] explicit size 24 GiB fits /dev/shm ...`
- present: `[kv-offload]   CPU tier policy=arc store_threshold=0`
- **absent:** any line mentioning `/kvcache` or `fs tier`

And the disk should stop growing:

```bash
df -h /kvcache          # run twice, minutes apart, under load — the used column should not move
```

### Undo it

Remove the `#`, restart, check for the orphan again. Nothing on disk changes format,
so the blocks already in `/kvcache` are still valid and will be picked up again —
provided `PYTHONHASHSEED` has not changed (it is pinned to 0 for exactly this reason).

### The one thing to remember

The **reaper stays mandatory as long as any blocks exist on disk**. The fs tier writes
and never deletes on its own. If you turn the tier off and later turn it back on, the
old blocks are still there and still counted. Leave `kvcache-reap.timer` enabled:

```bash
systemctl is-enabled kvcache-reap.timer     # want: enabled
```

If you are turning the disk tier off **permanently** and want the space back:

```bash
sudo -n systemctl stop llama-swap
sudo -n rm -rf /kvcache/blocks
sudo -n systemctl start llama-swap
```

Never delete `/kvcache/blocks` while the engine is running — a reaped in-flight block
kills EngineCore outright (there is no load-failure recovery path in this connector).

---

## 2. Raise `/dev/shm` to 32 GiB

`/dev/shm` is a **limit**, not an allocation. Raising it does not consume a byte on its
own; it only permits a bigger tier. It is safe and instant.

### Do it — live, no restart

```bash
sudo -n mount -o remount,size=32G /dev/shm
```

This is non-disruptive: it does not touch the region the running engine already
mapped, it only raises the ceiling. Verify:

```bash
python3 -c "import os;s=os.statvfs('/dev/shm');print('%.2f GiB'%(s.f_blocks*s.f_frsize/2**30))"
```

Expect `32.00 GiB`. **Do not trust `df -h` here** — it rounds the tmpfs up and has
misreported this number before.

### Make it survive a reboot

Edit `/etc/fstab` line 23. It currently reads:

```
tmpfs /dev/shm tmpfs rw,nosuid,nodev,inode64,size=28G 0 0
```

Change `size=28G` to `size=32G`. Back it up first, then verify the file still parses —
**an unparseable fstab can leave the box unbootable**, so never skip this check:

```bash
sudo -n cp /etc/fstab /etc/fstab.bak-$(date +%Y%m%d-%H%M%S)
sudo -n sed -i 's|\(tmpfs /dev/shm tmpfs .*\)size=28G|\1size=32G|' /etc/fstab
grep -n '/dev/shm' /etc/fstab          # read the line back and eyeball it
findmnt --verify                       # want: 0 parse errors (2 benign warnings are normal)
```

### Undo it

Restore the backup (`sudo -n cp /etc/fstab.bak-... /etc/fstab`) and
`sudo -n mount -o remount,size=28G /dev/shm`.

---

## 3. Set the KV tier to 30 GiB

**Read §3.0 before doing this.** It is the one change here that can hurt the box.

### 3.0 The honest sizing warning

The tier is **pre-faulted and pinned** — every byte is really committed at boot and
none of it can be swapped out to rescue you. So the tier size comes straight off the
top of the machine's RAM, permanently.

Measured on this box right now, with a 24 GiB tier:

```
MemTotal  39.17 GiB
used      31    GiB   (24 of which IS the tier)
available  7    GiB   ← everything the rest of the machine has to work with
```

Non-tier usage is therefore about **7 GiB**, and that is the number that matters.
Going 24 → 30 GiB takes 6 GiB of that 7:

| Tier | RAM left over | Verdict |
|---|---|---|
| 24 GiB (today) | ~7.0 GiB | known-good, running now |
| 26 GiB | ~5.0 GiB | comfortable |
| **28 GiB** | **~3.0 GiB** | **the largest I would run** — my recommendation |
| 30 GiB (what you asked for) | **~1.0 GiB** | works until it doesn't; a prefill spike OOM-kills the engine |

"No other services use this computer" is true and it is why 28 is even on the table —
but the ~7 GiB is not other services, it is **vLLM's own non-tier footprint**: weights
staging, the Python heap, CUDA/HIP host allocations, the page cache the O_DIRECT path
still needs around it. Those spike during prefill. At ~1 GiB of headroom there is no
margin for a spike, and the failure mode is not graceful degradation — it is the
kernel OOM-killer taking EngineCore, mid-request.

**My recommendation is 28 GiB with `/dev/shm` at 32 GiB.** That buys you 4 of the 6
extra GiB (about +127k tokens of tier) and keeps 3 GiB of headroom. The procedure
below does what you asked (30); to take the safer number instead, substitute 28 for 30
everywhere in §3.2 and use `KVOFF_RAM_RESERVE_GIB=11`.

### 3.1 The prerequisite: `/dev/shm` must be bigger than the tier

Do **§2 first**. A 30 GiB tier does not fit in a 28 GiB `/dev/shm` and the launcher
will refuse to boot (this guard is deliberate: asking for more than the tmpfs holds
does not degrade gracefully, it faults at `MADV_POPULATE_WRITE`). Keep the tmpfs
above the tier — 32 GiB tmpfs / 30 GiB tier leaves 2 GiB, which is the minimum I would
leave for podman's own locks and semaphores.

### 3.2 Do it — two lines must change together

**Line one — the tier size.** In `<your llama-swap config.yaml>`, in the `"qwen3.8-27b-vllm"` entry:

```yaml
      - "EXTRA=--kv-offloading-size 24 --kv-offloading-backend native"
```

becomes

```yaml
      - "EXTRA=--kv-offloading-size 30 --kv-offloading-backend native"
```

**Line two — the safety clamp, or line one does nothing.** The launcher has a
RAM-aware clamp that silently shrinks any request it thinks is unsafe. Its reserve
defaults to **15 GiB**, so it will allow at most `39.17 − 15 = 24.17` → **24 GiB**, and
your 30 would be clamped straight back to 24. You must lower the reserve in the same
edit. Add this line next to the others in the same entry:

```yaml
      - "KVOFF_RAM_RESERVE_GIB=9"
```

`39.17 − 9 = 30.17` → a 30 GiB request passes through untouched. (For the safer 28 GiB
tier, use `KVOFF_RAM_RESERVE_GIB=11`.)

**Do not just delete the clamp.** It is the only thing standing between a typo and an
unbootable box, and it is what correctly caught the old 31 GiB RAM configuration.
Lower it deliberately; leave it in place.

Then restart, and **check for the orphan** exactly as in §1:

```bash
sudo -n systemctl restart llama-swap
podman ps --format '{{.Names}}\t{{.Status}}'
```

### 3.3 Check it worked

Three things, in order. The boot log first:

```bash
sudo -n journalctl -u llama-swap --since "15 min ago" --no-pager | grep -i "kv-offload"
```

- **want:** `[kv-offload] explicit size 30 GiB fits /dev/shm (32.0 GiB) and RAM (39.17 GiB).`
- **do not want:** any line containing `*** CLAMPED` — if you see it, the reserve edit
  did not take, and the log tells you what it clamped to and why.

Then the region that actually got allocated. It appears several minutes into boot, so
wait for it:

```bash
until [ -n "$(ls /dev/shm/vllm_offload_*.mmap 2>/dev/null)" ]; do sleep 20; done
ls -la /dev/shm/vllm_offload_*.mmap | awk '{printf "%.2f GiB\n",$5/2**30}'
```

Expect **`29.99 GiB`** (the tier reports a hair under the request — 24 showed as 23.99).
Anything materially smaller means it was clamped or it did not fit.

Then the headroom you have left, which is the number to actually watch:

```bash
free -g | head -2
```

`available` should be **≥ 1 GiB**. If it is at or near 0, go back to 28 GiB — do not
wait to find out empirically.

### 3.4 Undo it

Put `--kv-offloading-size` back to 24, delete the `KVOFF_RAM_RESERVE_GIB` line,
restart, check for the orphan. Nothing about the cache's on-disk or in-tier format
depends on the size, so shrinking is clean — the tier just holds fewer blocks.

### 3.5 What you get for it

The tier holds roughly **31,700 tokens per GiB** (33,808 bytes/token measured — that is
the architectural floor for this model, 16 full-attention layers × 2,048 B/token, and
it is not reducible).

| Tier | Tokens | Prompts of ~34k |
|---|---|---|
| 16 GiB | ~508,000 | ~15 |
| 24 GiB (today) | ~762,000 | ~22 |
| 28 GiB | ~888,000 | ~26 |
| 30 GiB | ~951,000 | ~28 |

So 24 → 30 GiB is about **six more prompts** held in RAM instead of on disk. Worth
having, not worth an OOM.

---

## 4. Quick reference — all three, side by side

| Want | Change | Restart? | Risk |
|---|---|---|---|
| Disk tier off | comment out `KVOFF_DISK=/kvcache` | yes | none — pure capacity loss |
| `/dev/shm` → 32 GiB | `mount -o remount,size=32G /dev/shm` + `/etc/fstab` | no | none — it is only a limit |
| Tier → 30 GiB | `--kv-offloading-size 30` **and** `KVOFF_RAM_RESERVE_GIB=9` | yes | **OOM risk, ~1 GiB headroom** |

**After any `systemctl restart llama-swap`, check `podman ps` for an orphan.** The
restart signals the launcher, not the container; the container can outlive it and then
holds the GPU, the port and the `/dev/shm` region hostage.

---

## 5. If it will not boot after one of these

In order:

1. `sudo -n journalctl -u llama-swap --since "20 min ago" --no-pager | grep -i "kv-offload\|die\|error" | head -30`
2. `podman ps -a` — stop any orphan `qwen38-27b-vllm`.
3. `ls /dev/shm/` — a stale `vllm_offload_*.mmap` from a killed container holds its
   GiB until removed. The launcher reaps unreferenced ones at startup, but if one is
   stuck: confirm nothing holds it with `sudo -n fuser -v /dev/shm/vllm_offload_*.mmap`,
   then delete it.
4. Revert the edit (each section above has its undo), restart, confirm you are back.
5. Last resort — turn the offload off entirely by commenting out both the `EXTRA=`
   line's `--kv-offloading-size` and `KVOFF_DISK`. The model serves fine with no
   offload at all; it just recomputes more.

#!/usr/bin/env python3
"""Synthesise a /metrics fixture with the tier-report series present.

Lets tierreport.py be exercised end-to-end with no engine, no GPU and no
patched build:

    ./tier_report_fixture.py > /tmp/fixture.prom
    ./tierreport.py --metrics-file /tmp/fixture.prom

The numbers are invented but the SHAPES are real -- prometheus_client appends
_total to counters, histograms expose _bucket/_sum/_count with cumulative
counts, and the bucket edges are the ones patch_kv_offload_tier_report.py
declares. Anything the report gets wrong on this fixture it would get wrong on
a live endpoint.

Scenario, chosen so every verdict branch fires:
  cpu : oversized (65% of evictions never read) AND thrashing (p50 reuse 20 s)
  fs  : slower than recompute (break-even < 1.0) and net negative
"""
import sys
TR="vllm:kv_offload_tier_"
out=[]
def c(name, labels, v): out.append('%s_total{%s} %g' % (name, labels, v))
def g(name, labels, v): out.append('%s{%s} %g' % (name, labels, v))
def hist(name, labels, edges, counts, total_sum):
    cum=0
    for e,n in zip(edges,counts):
        cum+=n
        out.append('%s_bucket{%s,le="%s"} %g' % (name, labels, e, cum))
    out.append('%s_bucket{%s,le="+Inf"} %g' % (name, labels, cum))
    out.append('%s_sum{%s} %g' % (name, labels, total_sum))
    out.append('%s_count{%s} %g' % (name, labels, cum))

LAT=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,30,60,120,300]
# --- cpu: fast tier, 11 GB/s
cl='tier="cpu"'
c(TR+"hit_blocks", cl, 6000); c(TR+"hit_tokens", cl, 9_888_000)
c(TR+"load_bytes", cl, 336_192_000_000); c(TR+"load_seconds", cl, 30.0)
c(TR+"load_ops", cl, 6000)
c(TR+"store_bytes", cl, 400_000_000_000); c(TR+"store_seconds", cl, 60.0)
c(TR+"store_ops", cl, 7000)
hist(TR+"load_latency_seconds", cl, LAT, [0,0,0,1000,3000,1800,150,50,0,0,0,0,0,0,0], 30.0)
hist(TR+"store_latency_seconds", cl, LAT, [0,0,0,900,3500,2000,500,100,0,0,0,0,0,0,0], 60.0)
g(TR+"capacity_bytes", cl, 25_769_803_776); g(TR+"used_bytes", cl, 24_000_000_000)
hist(TR+"occupancy_ratio", cl, [0.5,0.75,0.9,0.95,0.98,0.99,0.999,1.0],
     [10,20,50,120,400,300,90,10], 950.0)
hist(TR+"reads_before_evict", cl, [0.5,1.5,2.5,4.5,8.5,16.5,32.5,64.5],
     [6500,1800,900,500,200,80,15,5], 6000.0)
hist(TR+"eviction_to_reuse_seconds", cl, [1,5,15,60,300,900,3600,14400,86400],
     [50,300,900,600,200,60,20,5,1], 120000.0)
c(TR+"evictions", cl, 10000); c(TR+"evicted_bytes", cl, 340_000_000_000)
c(TR+"lookup_miss_evicted", cl, 2136); c(TR+"stall_seconds", cl, 90.0)
# --- fs: 95 MB/s, slower than recompute; serves a little
fl='tier="fs"'
c(TR+"hit_blocks", fl, 400); c(TR+"hit_tokens", fl, 659_200)
c(TR+"load_bytes", fl, 22_412_800_000); c(TR+"load_seconds", fl, 236.0)
c(TR+"load_ops", fl, 400)
c(TR+"store_bytes", fl, 180_000_000_000); c(TR+"store_seconds", fl, 1800.0)
c(TR+"store_ops", fl, 3200)
hist(TR+"load_latency_seconds", fl, LAT, [0,0,0,0,0,0,20,150,180,45,5,0,0,0,0], 236.0)
hist(TR+"store_latency_seconds", fl, LAT, [0,0,0,0,0,0,100,1500,1300,280,20,0,0,0,0], 1800.0)
g(TR+"capacity_bytes", fl, 548_682_072_064); g(TR+"used_bytes", fl, 370_395_803_648)
hist(TR+"occupancy_ratio", fl, [0.5,0.75,0.9,0.95,0.98,0.99,0.999,1.0],
     [900,80,15,4,1,0,0,0], 350.0)
c(TR+"stall_seconds", fl, 640.0)
# --- engine-side, unpatched layer
for src,v in (("local_compute",21_316_676),("local_cache_hit",16_125_680),
              ("external_kv_transfer",11_011_936)):
    c("vllm:prompt_tokens_by_source", 'source="%s"' % src, v)
c("vllm:prompt_tokens", 'model_name="m"', 48_454_292)
out.append('vllm:request_prefill_time_seconds_sum{model_name="m"} 6100')
out.append('vllm:request_prefill_time_seconds_count{model_name="m"} 999')
out.append('vllm:request_prefill_kv_computed_tokens_sum{model_name="m"} 21316676')
out.append('vllm:request_prefill_kv_computed_tokens_count{model_name="m"} 999')
sys.stdout.write("\n".join(out)+"\n")

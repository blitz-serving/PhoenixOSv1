Build w/ CRIU. Just set BUILD_CRIU=1 and then build/run/test. 

```
BUILD_CRIU=1 cargo run -- ARGS
```

Test CPU-related CR with CRIU. 

```
cargo test --test cpucr
```
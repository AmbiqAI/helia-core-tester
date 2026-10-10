# Changelog

## [0.10.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.9.0...v0.10.0) (2026-10-10)


### Features

* **activation:** cover exact float16 GELU at ns-cmsis-nn v7.43.0 ([#450](https://github.com/AmbiqAI/helia-core-tester/issues/450)) ([bc5eaa2](https://github.com/AmbiqAI/helia-core-tester/commit/bc5eaa2119b0d6769bd0f3fceb0f56eab0321b64)), closes [#449](https://github.com/AmbiqAI/helia-core-tester/issues/449)
* **activation:** cover exact float32 GELU at ns-cmsis-nn v7.42.0 ([#448](https://github.com/AmbiqAI/helia-core-tester/issues/448)) ([7e1101b](https://github.com/AmbiqAI/helia-core-tester/commit/7e1101b07f2b3761f143de50c0999f5e0e3ede6f)), closes [#447](https://github.com/AmbiqAI/helia-core-tester/issues/447)
* **agent-loop:** add campaign workspace, judge and agent commands ([a565def](https://github.com/AmbiqAI/helia-core-tester/commit/a565defeaa1cad781a31bd132f6e7bdf98ca8543))
* **agent-loop:** chase speed before the size phase ([57496e9](https://github.com/AmbiqAI/helia-core-tester/commit/57496e9c66a2f025bd9624cc460663e986a2c0e4))
* **agent-loop:** chase speed before the size phase ([09f5811](https://github.com/AmbiqAI/helia-core-tester/commit/09f58113385d02c8c9695dcc0c65434c9ccf3f02))
* **agent-loop:** configurable kernel-optimization agent campaigns ([3c745d7](https://github.com/AmbiqAI/helia-core-tester/commit/3c745d7f265de6e450901583a2ce1c5cbf2fc8a9))
* **agent-loop:** judge candidates under gcc and atfe ([f293c78](https://github.com/AmbiqAI/helia-core-tester/commit/f293c782f5acba2f05cc901865fa41c4ad1a9f0b))
* **agent-loop:** run campaigns on s16 TransposeConv ([1c4a897](https://github.com/AmbiqAI/helia-core-tester/commit/1c4a897153c4c5413e5b2c8d53d4d973df17dde7))
* **agent-loop:** steer passing campaigns to shrink code ([e6a6e19](https://github.com/AmbiqAI/helia-core-tester/commit/e6a6e1967ba90340d056c9f79953f433eb735d8d))
* **agent-loop:** steer passing campaigns to shrink code ([af2a1a9](https://github.com/AmbiqAI/helia-core-tester/commit/af2a1a98375e1f853f7179094b5d739cf61d6b05))
* **conv:** call the per-route float convolution and depthwise entries directly ([#352](https://github.com/AmbiqAI/helia-core-tester/issues/352)) ([c2e1286](https://github.com/AmbiqAI/helia-core-tester/commit/c2e12861ea7b01acfa24af71cc6505272213cc29)), closes [#351](https://github.com/AmbiqAI/helia-core-tester/issues/351)
* **elementwise:** call the s8 row-broadcast add and mul entries directly ([#350](https://github.com/AmbiqAI/helia-core-tester/issues/350)) ([cd0102b](https://github.com/AmbiqAI/helia-core-tester/commit/cd0102b600d416ad96474ed658164a06fa145d55)), closes [#347](https://github.com/AmbiqAI/helia-core-tester/issues/347) [#343](https://github.com/AmbiqAI/helia-core-tester/issues/343)
* **generate:** seed hidden shapes from a secret outside the tree ([17cf84f](https://github.com/AmbiqAI/helia-core-tester/commit/17cf84f59f913cabf5a69aae3261f3a4076e9876))
* **generate:** seed hidden shapes from a secret outside the tree ([8add9ae](https://github.com/AmbiqAI/helia-core-tester/commit/8add9ae1dd5b7a853b731a0d5d485cb8e5aeb50f))
* **generation:** add MLPerf Tiny layer-shape kernel cases ([6a2bdd1](https://github.com/AmbiqAI/helia-core-tester/commit/6a2bdd155ecac6977b059b5104708ff85447dbcc))
* **generation:** add s16 TransposeConv cases for the direct kernel ([04b01f1](https://github.com/AmbiqAI/helia-core-tester/commit/04b01f119bc3559ef62de5cdb534a92136638529))
* **generation:** seeded random shapes for s8 conv and depthwise ([7ff41bf](https://github.com/AmbiqAI/helia-core-tester/commit/7ff41bf577f441c17eaa9b2d5b0bb0459f4fc6a7))
* **generation:** seeded random shapes for s8 conv and depthwise ([223e20a](https://github.com/AmbiqAI/helia-core-tester/commit/223e20a2799f3655caa585dec20f62f240f09eb6))
* **hardware:** add --toolchain gcc|atfe to build, flash and run ([3cb41d0](https://github.com/AmbiqAI/helia-core-tester/commit/3cb41d007b654645d85def014818a79eafee5d2f))
* **hardware:** add a no_gain verdict to score ([bc8be47](https://github.com/AmbiqAI/helia-core-tester/commit/bc8be470ffb0f64822aaa925a9fe9b0a63cff019))
* **hardware:** add an MRAM weights placement leg ([e368572](https://github.com/AmbiqAI/helia-core-tester/commit/e368572ffa91e5c0d954391992ddec15feb60584))
* **hardware:** add strict and self-golden compare modes ([c17b941](https://github.com/AmbiqAI/helia-core-tester/commit/c17b941c460c50273dc9d6824bd9274df5a9645f))
* **hardware:** bridge SVDF s8 and LSTM s8 cases to firmware ([77dfb23](https://github.com/AmbiqAI/helia-core-tester/commit/77dfb23c5d54323aa32cdce71753e1e88595ee1d))
* **hardware:** bridge SVDF s8 and LSTM s8 cases to firmware ([9cba00c](https://github.com/AmbiqAI/helia-core-tester/commit/9cba00c6bf1eedf52237626b3faee848333b8f50))
* **hardware:** candidate eval and baseline commands ([8c78f89](https://github.com/AmbiqAI/helia-core-tester/commit/8c78f8924bf37f6677fa72103a0fbfb4db540e8b))
* **hardware:** candidate eval and baseline commands ([ee53c31](https://github.com/AmbiqAI/helia-core-tester/commit/ee53c3121b11d5943a4b966281b972d8d60b8e37))
* **hardware:** default score --min-score to 0.005 ([c037541](https://github.com/AmbiqAI/helia-core-tester/commit/c037541fd676d0f314b714d4f9fea69d2fd3c35d))
* **hardware:** explain PMU counters per case ([0fff275](https://github.com/AmbiqAI/helia-core-tester/commit/0fff275b5e005ac17d432b410b787ce8fecbf6cb))
* **hardware:** explain PMU counters per case ([ed9b938](https://github.com/AmbiqAI/helia-core-tester/commit/ed9b93872e4b75839d4327ea66112ff5ab2212cc))
* **hardware:** explain why a case's cycles can't gate perf ([36593d7](https://github.com/AmbiqAI/helia-core-tester/commit/36593d7ca50e40ed0aeb75c2796c11e38ec40af3))
* **hardware:** focus, family gate and session bands in score ([288b980](https://github.com/AmbiqAI/helia-core-tester/commit/288b980e010142e433bc7505cbeb82fad11a4dd2))
* **hardware:** focus, family gate and session bands in score ([581dffc](https://github.com/AmbiqAI/helia-core-tester/commit/581dffcee82827284d0bd3a20b065dbe8e305cc7))
* **hardware:** gate cases only where kernel code changed ([46e8356](https://github.com/AmbiqAI/helia-core-tester/commit/46e8356beb05767ec85caf382bda8fcc641c6821))
* **hardware:** gate cases only where kernel code changed ([e8b227e](https://github.com/AmbiqAI/helia-core-tester/commit/e8b227ef7ad949ef7d6611d85447b9cdcaf7e080))
* **hardware:** generate, time and hidden-test arm_transpose_conv_s16 ([ffc8334](https://github.com/AmbiqAI/helia-core-tester/commit/ffc8334209a0488c9832b36911b70189c528a86e))
* **hardware:** lock the harness with a digest and candidate check ([ecbc919](https://github.com/AmbiqAI/helia-core-tester/commit/ecbc9190e6650beb2a506ec13b91d12fa2db8ab3))
* **hardware:** lock the harness with a digest and candidate check ([6a01b49](https://github.com/AmbiqAI/helia-core-tester/commit/6a01b49b3b983cb2f28e078c3dddc18fb535093d))
* **hardware:** name the inner kernel each s8 wrapper runs ([7524bf4](https://github.com/AmbiqAI/helia-core-tester/commit/7524bf4ae1e62765ac3ccead13cc941b1ca89a84))
* **hardware:** name the inner kernel each s8 wrapper runs ([689eb21](https://github.com/AmbiqAI/helia-core-tester/commit/689eb21807384c7e9f33e3a7f24f6b2489727d05))
* **hardware:** recheck the snapshot with built objects ([87efc06](https://github.com/AmbiqAI/helia-core-tester/commit/87efc0680c127b27ea8d0659eeaf1e8d4c7acbc8))
* **hardware:** report CMSIS-NN entry-point coverage per run ([4a7ef79](https://github.com/AmbiqAI/helia-core-tester/commit/4a7ef791656df86687908d35b90f30d036a8dcb1))
* **hardware:** report MACs, cycles per MAC and prepare cycles per case ([a359afb](https://github.com/AmbiqAI/helia-core-tester/commit/a359afb8ab8c6401255b598ea21a234081ed0459))
* **hardware:** run a hidden set beside the public cases ([be49294](https://github.com/AmbiqAI/helia-core-tester/commit/be4929464f21490cb919a7340f4ad52f5ad3211f))
* **hardware:** run a hidden set beside the public cases ([e2accc8](https://github.com/AmbiqAI/helia-core-tester/commit/e2accc88767ce39ef6c296cfadf2ac92c8ba8c54))
* **hardware:** scan built kernel objects in candidate check ([d5efcf3](https://github.com/AmbiqAI/helia-core-tester/commit/d5efcf32664c86fba21c4679dc4fd62b98384946))
* **hardware:** scan built kernel objects in candidate check ([155e1c7](https://github.com/AmbiqAI/helia-core-tester/commit/155e1c765eeec7789ba1ab545cb561aa015875b9))
* **hardware:** score hidden cases apart, refuse mismatched sets ([f992bd1](https://github.com/AmbiqAI/helia-core-tester/commit/f992bd10041036c5f0d4e3d67aa16bd7630b154a))
* **hardware:** score hidden cases apart, refuse mismatched sets ([3250346](https://github.com/AmbiqAI/helia-core-tester/commit/325034663357e2b6387b4a7a7610432b01a6d59c))
* **hardware:** score kernel candidates against a baseline ([c33672a](https://github.com/AmbiqAI/helia-core-tester/commit/c33672ae4201cc98d582c7341af3a47c92ec9600))
* **hardware:** score kernel candidates against a baseline ([fb8b275](https://github.com/AmbiqAI/helia-core-tester/commit/fb8b275806f4c7a94abc796e9c2653bba8d0a7fb))
* **hardware:** split depthwise opt route; fill s4/s16 routes ([c5d1bb0](https://github.com/AmbiqAI/helia-core-tester/commit/c5d1bb06fc66bfa2c9e071e955fa8e4ca5147c26))
* **hardware:** split depthwise opt route; fill s4/s16 routes ([33550e4](https://github.com/AmbiqAI/helia-core-tester/commit/33550e46160745f0fb2ef0d3f3e4b84b7b808c96)), closes [#376](https://github.com/AmbiqAI/helia-core-tester/issues/376)
* **hardware:** time arm_transpose_conv_s16 on the board ([8f43f43](https://github.com/AmbiqAI/helia-core-tester/commit/8f43f43d38eb248f8e4b3c04655304f8a068fe18))
* **hardware:** time s8 direct-entry cases on the board ([a45f96e](https://github.com/AmbiqAI/helia-core-tester/commit/a45f96e46b32961e2006d3fccd2400fe3db2dae3))
* **hardware:** time s8 direct-entry cases on the board ([c88771a](https://github.com/AmbiqAI/helia-core-tester/commit/c88771a961f4aec7e27eed1e3c0e8c72d098ab3d))
* **hardware:** time the s8 conv and depthwise wrappers ([faac988](https://github.com/AmbiqAI/helia-core-tester/commit/faac98803b3b4f31d5396c8067669e13d72b71c8))
* **hardware:** write harness_digest at the manifest top level ([dfebcda](https://github.com/AmbiqAI/helia-core-tester/commit/dfebcdaf397a047afdd5257f71ab9a33e3874ef8))
* **random-shapes:** add hidden s16 DepthwiseConv shapes ([1c71df9](https://github.com/AmbiqAI/helia-core-tester/commit/1c71df95c0429b3415829f308694fd2ae4c4b121))
* **random-shapes:** add hidden s16 TransposeConv shapes ([22dc531](https://github.com/AmbiqAI/helia-core-tester/commit/22dc5315fea46d478a0dfbdea08e4b9bb5f29e08))
* **random-shapes:** per-op random and hidden shape draws ([ee038b9](https://github.com/AmbiqAI/helia-core-tester/commit/ee038b9f4c0d7f7c538d25025cdca61a19b4284c))
* **random-shapes:** select generators per op and dtype ([c3e4c74](https://github.com/AmbiqAI/helia-core-tester/commit/c3e4c74d08a15b688b893070608876a1b6e448ac))
* **score:** gate and score touched cases only ([1c91e3b](https://github.com/AmbiqAI/helia-core-tester/commit/1c91e3be3fb0e0e5f1079e10962ace80ad7232fe))
* **score:** gate and score touched cases only ([eb441c9](https://github.com/AmbiqAI/helia-core-tester/commit/eb441c9f52acf1d618426fa496b1da5a9aef4dae))


### Bug Fixes

* **agent-loop:** bound submit lock, verify clones, unique selftest probes ([06604de](https://github.com/AmbiqAI/helia-core-tester/commit/06604deabe9f302e948035d6ae80c2e90f8f48eb))
* **agent-loop:** clear ready while init reruns ([4763dcb](https://github.com/AmbiqAI/helia-core-tester/commit/4763dcb47cba77caad481994bd6ce27c25d51834))
* **agent-loop:** compact status rows, document measured timings ([4a5ed9c](https://github.com/AmbiqAI/helia-core-tester/commit/4a5ed9c7b902bbd273d0e9e902178d73779d74f7))
* **agent-loop:** drop a no-gain smallest from next ([942bd71](https://github.com/AmbiqAI/helia-core-tester/commit/942bd7166c7536e0b77c1599fd93e127d2c17ca7))
* **agent-loop:** freeze submitted trees, charge kernel faults, resume init ([1b3edb5](https://github.com/AmbiqAI/helia-core-tester/commit/1b3edb52f1d463157a62c01d42cbcceab094fb90))
* **agent-loop:** generate hidden shapes against the base tree ([9698921](https://github.com/AmbiqAI/helia-core-tester/commit/969892127814e60d5eaed3c2af78ddfb47a26ded))
* **agent-loop:** harden resume, per-run logs, init marker, deadlines ([57115d8](https://github.com/AmbiqAI/helia-core-tester/commit/57115d8424ebcdb7b027452fb57cc2108e356b3a))
* **agent-loop:** keep baselines unless both compilers are known ([f07eb11](https://github.com/AmbiqAI/helia-core-tester/commit/f07eb11315244d0057faa7004d79488b3ba88c5d))
* **agent-loop:** keep legacy disasm call, guide atfe-only prompts ([e28e61f](https://github.com/AmbiqAI/helia-core-tester/commit/e28e61f5b48f7567cbebce2bf8f8f74b569dd1d5))
* **agent-loop:** keep toolchain facts in campaign order ([374fb15](https://github.com/AmbiqAI/helia-core-tester/commit/374fb15ec5ea14d28ece26d80e19e1e55acbacba))
* **agent-loop:** parse start patches strictly, own the secrets dir ([92e243c](https://github.com/AmbiqAI/helia-core-tester/commit/92e243c91ee422f58f99055d7f302abca6fc0c54))
* **agent-loop:** rank fastest pass without size, tidy prompt ([05ee339](https://github.com/AmbiqAI/helia-core-tester/commit/05ee339c068678f07c050db8d3a89f08a0278b41))
* **agent-loop:** trust only ATFE_ROOT clang ([2b2584f](https://github.com/AmbiqAI/helia-core-tester/commit/2b2584fd1057922466bef68059e0a13080d6e348))
* **check:** skip gcc's cwd marker in scan deps ([20b0fa6](https://github.com/AmbiqAI/helia-core-tester/commit/20b0fa6be5ab61b5673c5a1ec71dccd1c0fc5088))
* **check:** skip gcc's cwd marker in scan deps ([7942623](https://github.com/AmbiqAI/helia-core-tester/commit/7942623aff802b60e6d7fc1f19c0fdb02213a9bc))
* **config:** bound shape_seed to 32 bits so case ids fit ([06b75f9](https://github.com/AmbiqAI/helia-core-tester/commit/06b75f9882fd1bd51eef3ae6df754daf01323e00))
* **conv:** compute FP16 Convolve goldens from the float16 weights the kernel receives ([#346](https://github.com/AmbiqAI/helia-core-tester/issues/346)) ([79d0832](https://github.com/AmbiqAI/helia-core-tester/commit/79d083256edd586572bae081e4a95a7bcd11de08)), closes [#335](https://github.com/AmbiqAI/helia-core-tester/issues/335)
* **generate:** derive every hidden output from DIR, check DIR both ways ([bc3484e](https://github.com/AmbiqAI/helia-core-tester/commit/bc3484e906631eaa0bd0c0a5571c776ea37b001a))
* **generate:** guard direct hidden runs and stray seed files ([883f622](https://github.com/AmbiqAI/helia-core-tester/commit/883f6227f6d91d3a6674363991751f69a0293398))
* **generate:** keep hidden output and seed out of the tree ([7eaf2b1](https://github.com/AmbiqAI/helia-core-tester/commit/7eaf2b1d2316b99dfd640b7f2a46d2383564d046))
* **generate:** refuse any symlink in hidden trees, never follow links when pruning ([e6481fa](https://github.com/AmbiqAI/helia-core-tester/commit/e6481fa74624102d8f146385927155af5bc44eb0))
* **generate:** refuse hard links and any shape seed in hidden mode ([46fa49a](https://github.com/AmbiqAI/helia-core-tester/commit/46fa49ac3e17a5aeb23f4b7d1730ccefac0cc050))
* **generate:** refuse hidden destinations that symlink out of DIR ([458ed06](https://github.com/AmbiqAI/helia-core-tester/commit/458ed0696a0017d139b023f6d36b93d7f4044b2b))
* **generation:** compute FP16 depthwise, transpose and fully connected goldens from float16 weights ([#355](https://github.com/AmbiqAI/helia-core-tester/issues/355)) ([f9d2326](https://github.com/AmbiqAI/helia-core-tester/commit/f9d23267483d315921dde1d8d08f4d18fd64d7b5)), closes [#345](https://github.com/AmbiqAI/helia-core-tester/issues/345)
* **generation:** convert batch-2 s16 transpose conv with a fixed batch ([15ae384](https://github.com/AmbiqAI/helia-core-tester/commit/15ae384ee395553420f22832613f977d9e58a027))
* **generation:** convert batch-2 s16 transpose conv with a fixed batch ([543a373](https://github.com/AmbiqAI/helia-core-tester/commit/543a37377976ab59fbe4e6586e00b9755a1d77fc))
* **generation:** fail when any --op value matches no descriptor ([a1df97d](https://github.com/AmbiqAI/helia-core-tester/commit/a1df97d134d65c656a6ae7de3eee0cc0b8a9de08))
* **generation:** fail when any --op value matches no descriptor ([49d55e3](https://github.com/AmbiqAI/helia-core-tester/commit/49d55e3b4f704dd7964993c54b8f15a6fcdf7305)), closes [#379](https://github.com/AmbiqAI/helia-core-tester/issues/379)
* **generation:** gate 3x3 entry cases on their sizer ([bb81a52](https://github.com/AmbiqAI/helia-core-tester/commit/bb81a5238d12a55e0ab8c4bdf3a136888066e45a))
* **generation:** keep unselected cases on forced runs, drop stale selected ([15b0cf4](https://github.com/AmbiqAI/helia-core-tester/commit/15b0cf4e5d868b55f982044d857e7f937a8abf50))
* **generation:** key MLPerf layer shapes on activation too ([5cdade6](https://github.com/AmbiqAI/helia-core-tester/commit/5cdade6006bb369b5abdbe4d4748d5e37cfbe4ee))
* **generation:** key MLPerf layer shapes on activation too ([8bd620c](https://github.com/AmbiqAI/helia-core-tester/commit/8bd620c0df2b82a1ac104f54d98d0e76972b4618))
* **generation:** pin comparison ties under two-sided broadcast ([968e26e](https://github.com/AmbiqAI/helia-core-tester/commit/968e26e54ee14bcd0192ce0894d7a4f233022112))
* **generation:** rescale-aware comparison gap, sub and mid rsqrt ([c19e8a8](https://github.com/AmbiqAI/helia-core-tester/commit/c19e8a846eed202bd026201035bfd946af62d697))
* **generation:** stop goldens collapsing to constant or saturated values ([a0ccf3a](https://github.com/AmbiqAI/helia-core-tester/commit/a0ccf3ae8b17fbdb7535eee79853db9066a8717c))
* **generation:** validate random-shape settings, share targets ([c56d335](https://github.com/AmbiqAI/helia-core-tester/commit/c56d335d61e529e55f397141a4284696aeb9a0d3))
* **generation:** widen s8 binary, comparison and rsqrt inputs ([4170292](https://github.com/AmbiqAI/helia-core-tester/commit/41702927caf296b913bf2464bfd106ec8402e980))
* **generation:** widen s8 binary, comparison and rsqrt inputs ([74f03cc](https://github.com/AmbiqAI/helia-core-tester/commit/74f03cc67b02065dd4d12b8bf36430c20cd19b41))
* **hardware:** address explain review findings ([8355460](https://github.com/AmbiqAI/helia-core-tester/commit/8355460b771f64b379cc0a71123644197d57f6eb))
* **hardware:** allow stricter candidates, require baseline goldens ([8a0f516](https://github.com/AmbiqAI/helia-core-tester/commit/8a0f516ba6d64da71a09da22b3337f5c79b6b067))
* **hardware:** alternate input copies and check read-only operands ([b48ec08](https://github.com/AmbiqAI/helia-core-tester/commit/b48ec088758c0d2fdb86e42e9079b1036b783168))
* **hardware:** alternate input copies and check read-only operands ([b3bf28f](https://github.com/AmbiqAI/helia-core-tester/commit/b3bf28f19791a176b161fe6c673f8fface4eb537))
* **hardware:** bound binutil output in the object scan ([c443214](https://github.com/AmbiqAI/helia-core-tester/commit/c4432144bc86111a3f7efdd81709ca8e98499f08))
* **hardware:** bound object sizes and object scan time ([39ba6f9](https://github.com/AmbiqAI/helia-core-tester/commit/39ba6f961770ee9c7b367aa793f4f8f1ca855836))
* **hardware:** bound the snapshot copy, refuse all-missing runs ([100967f](https://github.com/AmbiqAI/helia-core-tester/commit/100967f4c15e0d529eb17f9b06fcfcdade733bc6))
* **hardware:** bound the whole gcc -E scan ([d77dd1b](https://github.com/AmbiqAI/helia-core-tester/commit/d77dd1bea9b8d1ab20c0a32382c32a44bb92932e))
* **hardware:** bridge the hidden set once, before generation ([514bceb](https://github.com/AmbiqAI/helia-core-tester/commit/514bcebed7ce60fd6898cbd69d32966d29216b33))
* **hardware:** candidate eval exits 130 on Ctrl-C ([2b18286](https://github.com/AmbiqAI/helia-core-tester/commit/2b18286eb33d35a8764db0477e6a35c3200824fa))
* **hardware:** cap scan workers by a memory budget ([d8adfda](https://github.com/AmbiqAI/helia-core-tester/commit/d8adfda178a861583a87e441fa62a809507ce474))
* **hardware:** catch digraph directives and asm IRQ masking, hash module trees ([3c5530d](https://github.com/AmbiqAI/helia-core-tester/commit/3c5530d40d03bf3c0406ad6d88da8f104b18dded))
* **hardware:** charge dirs, depth and copied bytes in the snapshot ([23c139a](https://github.com/AmbiqAI/helia-core-tester/commit/23c139a873fad4265f33c882f20d96557db9df91))
* **hardware:** check hidden cases before flashing, skip their FVP gate ([827dddf](https://github.com/AmbiqAI/helia-core-tester/commit/827dddf2aef22b07299cf1457d9350cc63935192))
* **hardware:** check inputs in --golden-from ([e0ec67a](https://github.com/AmbiqAI/helia-core-tester/commit/e0ec67a883c1b730f6140ff1e79b5712d3b4f9de))
* **hardware:** check inputs in --golden-from ([76589e0](https://github.com/AmbiqAI/helia-core-tester/commit/76589e04578ed2b41c76dd5fde07aa99131345ad))
* **hardware:** check SVDF and LSTM extents the kernels read ([d97d7a8](https://github.com/AmbiqAI/helia-core-tester/commit/d97d7a8ad8bb9247611a27dccd22e18eca381b4b))
* **hardware:** close candidate check bypasses, hash modules.cmake ([67b3d3c](https://github.com/AmbiqAI/helia-core-tester/commit/67b3d3c3398090e7ce787f6919aae123fdb9f159))
* **hardware:** close candidate check review gaps ([74a4cf2](https://github.com/AmbiqAI/helia-core-tester/commit/74a4cf25e57c688bdc49ab2abf772df37c222c71))
* **hardware:** copy candidate trees by dir fd, skip special files ([c666e5d](https://github.com/AmbiqAI/helia-core-tester/commit/c666e5d4eebfc51db72a9122490be4dc5c7d1076))
* **hardware:** count skip-worktree and assume-unchanged edits as dirty ([48fb33f](https://github.com/AmbiqAI/helia-core-tester/commit/48fb33f6fd57ad89c400bea44a64be63291ae153))
* **hardware:** distinct exit codes for refusals, flat explain cases ([33f9bfe](https://github.com/AmbiqAI/helia-core-tester/commit/33f9bfec863bc26bf2c51bfcdad406c2b64d91d4))
* **hardware:** distinct exit codes for refusals, flat explain cases ([20c7264](https://github.com/AmbiqAI/helia-core-tester/commit/20c7264d54ad0b0c21c94bbab888a94f1bc5edc5))
* **hardware:** drop stale module files, hash root manifest ([32e4ffe](https://github.com/AmbiqAI/helia-core-tester/commit/32e4ffe5b420f3cfeb3fb24568c0e0b7ed6ea0ba))
* **hardware:** emit MRAM capability flag from catalog generator ([ace2dbc](https://github.com/AmbiqAI/helia-core-tester/commit/ace2dbc81e3d1c386804ef62199687b108be058f))
* **hardware:** emit MRAM capability flag from catalog generator ([4c8617c](https://github.com/AmbiqAI/helia-core-tester/commit/4c8617c02677b11861253e80ece670fa4559a76f))
* **hardware:** eval a snapshot, never git in the agent repo ([b785767](https://github.com/AmbiqAI/helia-core-tester/commit/b78576723c746fbc5d492bf8e6fc82046b7a0216))
* **hardware:** every eval path prints a verdict; check placement ([2089fdc](https://github.com/AmbiqAI/helia-core-tester/commit/2089fdcf783ab64783f767ff70893cae8826cf73))
* **hardware:** exit 130 on Ctrl-C, 5 on abort ([fa8d5fe](https://github.com/AmbiqAI/helia-core-tester/commit/fa8d5fe5ddcdce355c4563de6f26d34e09373b84))
* **hardware:** exit 5 for bugs anywhere in a hardware command ([e7142a8](https://github.com/AmbiqAI/helia-core-tester/commit/e7142a8feb37d244bdd86f06e55a0885d32e916f))
* **hardware:** exit 5 on unexpected tester errors ([84a913b](https://github.com/AmbiqAI/helia-core-tester/commit/84a913b078f7fcf19db170e89ed391be6ff6fc6b))
* **hardware:** extract base tar without extractall filter ([b5e364f](https://github.com/AmbiqAI/helia-core-tester/commit/b5e364fca67bbe573475c451e11908a6717b8508))
* **hardware:** fail lost timing and judge bands by baseline ([4c18e4b](https://github.com/AmbiqAI/helia-core-tester/commit/4c18e4bff1718e5f975a2bfb74de1196e23ff564))
* **hardware:** fail prepare growth only when it buys a gain ([73cd6f1](https://github.com/AmbiqAI/helia-core-tester/commit/73cd6f158a3461941a37e20e54102175e14fd431))
* **hardware:** fail prepare growth only when it buys a gain ([4647da3](https://github.com/AmbiqAI/helia-core-tester/commit/4647da33a108ac03e22e5629e0e39eeb76ef7cd6))
* **hardware:** flag code and branch targets outside .text ([761e93a](https://github.com/AmbiqAI/helia-core-tester/commit/761e93a659b40d5b02d7967ca440efffb430408a))
* **hardware:** flag macro edits any conditional can reach ([916562a](https://github.com/AmbiqAI/helia-core-tester/commit/916562a7185ae1ba4eed91598fc5a725b9690c28))
* **hardware:** flag removed guards near forbidden constructs ([cecae3e](https://github.com/AmbiqAI/helia-core-tester/commit/cecae3ed9f7111d9267419db8300fa9caf8f3262))
* **hardware:** fold duplicate symbols by unit and binding ([d8a7e51](https://github.com/AmbiqAI/helia-core-tester/commit/d8a7e514c3321a3b4df2db67f635e8adb742c986))
* **hardware:** freeze candidate closure, splice include lines ([0ad4079](https://github.com/AmbiqAI/helia-core-tester/commit/0ad4079b435d5050e8d3eef64a3ab479b15c9e93))
* **hardware:** gate focused and unfocused family subsets apart ([89a3d6b](https://github.com/AmbiqAI/helia-core-tester/commit/89a3d6bdfd58167a14c5f3336780061baa12fe66))
* **hardware:** gate valid_for_regression on correct output ([eae74bb](https://github.com/AmbiqAI/helia-core-tester/commit/eae74bbdec0aff79771f59f4c97dab169cb3ef11))
* **hardware:** gate valid_for_regression on correct output ([75fdbc6](https://github.com/AmbiqAI/helia-core-tester/commit/75fdbc695b4528fb9eabf600403d5eb99b231b47))
* **hardware:** give depthwise its own inst/MAC target ([d891191](https://github.com/AmbiqAI/helia-core-tester/commit/d8911914fea6d5d98a8a04ffe228d868a53c6ba1))
* **hardware:** give depthwise its own inst/MAC target ([5fbc815](https://github.com/AmbiqAI/helia-core-tester/commit/5fbc81539b369d85e6d350a431382c4692286803))
* **hardware:** harden candidate check and harness digest ([0529ccc](https://github.com/AmbiqAI/helia-core-tester/commit/0529ccc2caf9671893c11ac445f11abcbbeb2c77))
* **hardware:** join %:%: digraph pastes in candidate rules ([88b2ca6](https://github.com/AmbiqAI/helia-core-tester/commit/88b2ca6f58b3761f1b6d6240e43b2c4fb73db777))
* **hardware:** join adjacent string literals, flag hidden index entries ([9355f14](https://github.com/AmbiqAI/helia-core-tester/commit/9355f145137ccaf592b81069b9893ce4b9196b3a))
* **hardware:** keep degenerate goldens in the perf gate ([f62cf9a](https://github.com/AmbiqAI/helia-core-tester/commit/f62cf9ab3aaca56c018065872635e6370a54bb4a))
* **hardware:** keep gcc -E text out of the parent ([50b2327](https://github.com/AmbiqAI/helia-core-tester/commit/50b232748c82b5c9c639e98736286e1f5750ff2f))
* **hardware:** keep input twin alignment; neutral operand reason ([0226db6](https://github.com/AmbiqAI/helia-core-tester/commit/0226db6746e90053f969d8691858a62363df493b))
* **hardware:** keep MRAM placement flag in generated catalog ([d3a5383](https://github.com/AmbiqAI/helia-core-tester/commit/d3a53835b94c08324810ae42f162dcd066886dee))
* **hardware:** keep other cases when a run narrows generation ([7fdfc84](https://github.com/AmbiqAI/helia-core-tester/commit/7fdfc8482492e1a1cd1155f9c2c6fc06116d7611))
* **hardware:** keep run options beside the digest, tighten pragma and guard checks ([40092b5](https://github.com/AmbiqAI/helia-core-tester/commit/40092b58f1e00120a073d6b809a697d444c693a7))
* **hardware:** keep run refusals, share the copy budget, count hidden ids ([3542a3e](https://github.com/AmbiqAI/helia-core-tester/commit/3542a3e37c78f87c22e4804f59001f2044d7908e))
* **hardware:** keep the handshake in the first batch trace ([f6e8710](https://github.com/AmbiqAI/helia-core-tester/commit/f6e8710cf6e12204e62242d8b641b54ba3170656))
* **hardware:** keep the header closure inside Include/ ([387dc15](https://github.com/AmbiqAI/helia-core-tester/commit/387dc156b8c49a4eb4293f3481036c41f69186d1))
* **hardware:** lex comments before include scan, survive link loops ([9ef27cf](https://github.com/AmbiqAI/helia-core-tester/commit/9ef27cf5207dadf074e7a2e567b18cc799d6e79c))
* **hardware:** lock kernel build files and harness headers ([29c5065](https://github.com/AmbiqAI/helia-core-tester/commit/29c5065cfc3f21adb25dbbfb4d6bbcec228892ef))
* **hardware:** lock kernel build files and harness headers ([04babf0](https://github.com/AmbiqAI/helia-core-tester/commit/04babf07a31809be1d90ad3a6f51d370a10d54cc))
* **hardware:** lock module swaps, root the float headers ([2975262](https://github.com/AmbiqAI/helia-core-tester/commit/29752621e6f548615752b573a3b01c7d3e422d0b))
* **hardware:** lock only where fcntl exists; accept VT/FF in directives ([4eb2e1f](https://github.com/AmbiqAI/helia-core-tester/commit/4eb2e1f81844738c81a00c3c7ba670f452448ee5))
* **hardware:** match one case dtype and narrow run generation ([5678613](https://github.com/AmbiqAI/helia-core-tester/commit/5678613fe6aa250fe5ab6b631e142f6cd2bc5c8c))
* **hardware:** match one case dtype and narrow run generation ([7dfcbc6](https://github.com/AmbiqAI/helia-core-tester/commit/7dfcbc6933d0e009f93634b713764adfe6e8a795))
* **hardware:** match SVDF kernel-sum width to the input ([c0be393](https://github.com/AmbiqAI/helia-core-tester/commit/c0be3935beb8f1a6373f757650347f994804a30c))
* **hardware:** match unsafe git config keys whole, any case ([7b98d5a](https://github.com/AmbiqAI/helia-core-tester/commit/7b98d5a7c8872e9e3b74ae1ae5ce0d62d9bf41e3))
* **hardware:** name the MVE multiply ratio for what the event counts ([e317b98](https://github.com/AmbiqAI/helia-core-tester/commit/e317b980249ebbd4a9c9028e01638d8446d0c028))
* **hardware:** need memory evidence for memory_bound, skip unmeasured clauses ([a5aa5bd](https://github.com/AmbiqAI/helia-core-tester/commit/a5aa5bd105b83ce4d0e8e41d3591301d6eec4643))
* **hardware:** never follow candidate or stale module links ([351f3c3](https://github.com/AmbiqAI/helia-core-tester/commit/351f3c3ec86be40f22651d189eecf855b0a1a7d0))
* **hardware:** never open special or oversized candidate files ([6040557](https://github.com/AmbiqAI/helia-core-tester/commit/60405571e9d46002da539f11db40680c316603d3))
* **hardware:** parse git config keys, refuse empty driver names ([f59895e](https://github.com/AmbiqAI/helia-core-tester/commit/f59895e6ab8d1bb33e110502f3a605f6ae585c93))
* **hardware:** pass the check to score, hash the snapshot ([ed03ad8](https://github.com/AmbiqAI/helia-core-tester/commit/ed03ad872ed412f2ac03a7d6f2ffd6ce69d94e7b))
* **hardware:** per-unit build flags, strong-global symbol check ([2e622cb](https://github.com/AmbiqAI/helia-core-tester/commit/2e622cb10e0c782b9d1aaf6b74af5b90d0940127))
* **hardware:** pin FPSCR and record it in TARGET_INFO ([15a5857](https://github.com/AmbiqAI/helia-core-tester/commit/15a585722f453778450d09bfd32eb81a7db619d1))
* **hardware:** pin kernel function alignment ([af241a2](https://github.com/AmbiqAI/helia-core-tester/commit/af241a254d666cee034f30c6a3b716e1eecf7e1a))
* **hardware:** pin kernel function alignment ([1276c58](https://github.com/AmbiqAI/helia-core-tester/commit/1276c58cd86bd1df96d57c429e465b95d79cdccb))
* **hardware:** pin ns-cmsis-nn v7.40.0 ([#371](https://github.com/AmbiqAI/helia-core-tester/issues/371)) ([958b790](https://github.com/AmbiqAI/helia-core-tester/commit/958b7901edabcbf77e9d49e76670fe205aa7fa12)), closes [#356](https://github.com/AmbiqAI/helia-core-tester/issues/356)
* **hardware:** poison output before timed calls and check it ([eb77371](https://github.com/AmbiqAI/helia-core-tester/commit/eb77371efd1112ac168bcb73e00ae58c2ba8d351))
* **hardware:** poison output before timed calls and check it ([fcafca9](https://github.com/AmbiqAI/helia-core-tester/commit/fcafca9a9664a886cdca5bb1bd129b4b408817da))
* **hardware:** reach wrapper helpers, hash alignment ([9e222cd](https://github.com/AmbiqAI/helia-core-tester/commit/9e222cd269b8036f13333d1039c182624988527c))
* **hardware:** read object sections from the ELF table ([41d0f95](https://github.com/AmbiqAI/helia-core-tester/commit/41d0f95e4ad32152bd4ba0df10ef5641b29394ac))
* **hardware:** record no output digest for status-only cases ([9c5a708](https://github.com/AmbiqAI/helia-core-tester/commit/9c5a708ef7e2def2ac27fa99f6d06c11102bd038))
* **hardware:** refuse a bad hidden set with exit 3 ([d4f7154](https://github.com/AmbiqAI/helia-core-tester/commit/d4f7154324de00e8a4ab39c08f7d9d0a66d43c63))
* **hardware:** refuse a dirty tester and missing bundles up front ([0060740](https://github.com/AmbiqAI/helia-core-tester/commit/00607402ccc305ebb8b1531b0e82b404a0519b37))
* **hardware:** refuse candidate repos whose git config runs code ([87e6f07](https://github.com/AmbiqAI/helia-core-tester/commit/87e6f07cb0fa774e814eca83dca06f0ab8888b15))
* **hardware:** refuse candidate repos whose git config runs code ([9d2ee86](https://github.com/AmbiqAI/helia-core-tester/commit/9d2ee86264a7df05e3a6e4df2700ef9dbe935012))
* **hardware:** refuse every pre-run misfit with exit 3 ([24e5aff](https://github.com/AmbiqAI/helia-core-tester/commit/24e5aff7869394cb9416e4749350826e217e878f))
* **hardware:** refuse mismatched build flags on stream-only runs ([c44a055](https://github.com/AmbiqAI/helia-core-tester/commit/c44a05535f0befe29919944d88ade29d6cd1e8a3))
* **hardware:** refuse non-finite cycles and settings, compare digests per side ([13533ac](https://github.com/AmbiqAI/helia-core-tester/commit/13533ac9678e86d412cc552ea516bb69f7aa008d))
* **hardware:** refuse unknown kernel repeats, exclude partial baselines ([b436432](https://github.com/AmbiqAI/helia-core-tester/commit/b436432e6fc616167e1970670b72b20d6e27d86f))
* **hardware:** report a broken descriptor catalog cleanly ([459f8a4](https://github.com/AmbiqAI/helia-core-tester/commit/459f8a459cc89e74009d136c40eab91318a680cc))
* **hardware:** report a disjoint golden run as missing cases ([99125bd](https://github.com/AmbiqAI/helia-core-tester/commit/99125bdc0e643f3c64822df054a9a55035e27747))
* **hardware:** report pct_of_peak in percent ([787cf3e](https://github.com/AmbiqAI/helia-core-tester/commit/787cf3e4b987a7fd2575390f16792811e4f27ea4))
* **hardware:** report pct_of_peak in percent ([fe54fa8](https://github.com/AmbiqAI/helia-core-tester/commit/fe54fa862059cb4dc647d0c72adc3df791cf7fa0))
* **hardware:** require a clean HEAD for local-root kernels ([049a1d1](https://github.com/AmbiqAI/helia-core-tester/commit/049a1d1efccac3c1abe90587521fd05c635bc30a))
* **hardware:** require a full base SHA, count rule hits per whole file ([febb619](https://github.com/AmbiqAI/helia-core-tester/commit/febb6196cd4582d2906afad16f8fcbea946e26b2))
* **hardware:** rerun candidate rules on gcc -E output ([c6b1c27](https://github.com/AmbiqAI/helia-core-tester/commit/c6b1c27a9906ac8fb3b2ef2ab844ba67c6addfa5))
* **hardware:** rerun candidate rules on gcc -E output ([abbbe41](https://github.com/AmbiqAI/helia-core-tester/commit/abbbe413e9853f977f2471bb9a856ff763692f70))
* **hardware:** reserve im2col scratch for every S4 1xN convolve ([58500c0](https://github.com/AmbiqAI/helia-core-tester/commit/58500c05735c029072c5270334a961e9eb3da3cb))
* **hardware:** reserve im2col scratch for every S4 1xN convolve ([c50f80a](https://github.com/AmbiqAI/helia-core-tester/commit/c50f80a7eb36511705b029581a1642b563b75931))
* **hardware:** reset unpassed build switches to defaults ([f92c79f](https://github.com/AmbiqAI/helia-core-tester/commit/f92c79f34062cd99cc9ea594ce079fe7417e8f58))
* **hardware:** reset unpassed build switches to defaults ([f31273d](https://github.com/AmbiqAI/helia-core-tester/commit/f31273ddf4f84fa4bb42086bb56be1917e98fa2c))
* **hardware:** restore the old module when the swap fails ([64ae183](https://github.com/AmbiqAI/helia-core-tester/commit/64ae183f259927d45c7ba97257942d62d20f8024))
* **hardware:** review fixes for the route split ([45085b9](https://github.com/AmbiqAI/helia-core-tester/commit/45085b91f19aa759e1178a0de305b9bc92584e40))
* **hardware:** route by the built kernel tree and firmware defaults ([e47a848](https://github.com/AmbiqAI/helia-core-tester/commit/e47a84815bdc5a9526d6ca424b4c475a0555ef98))
* **hardware:** run candidate rules on one phase 1-3 view ([00fee16](https://github.com/AmbiqAI/helia-core-tester/commit/00fee16a5e9c5b945ea83677d142b851941b7b19))
* **hardware:** share predicated cycles by cycles, note s16 depthwise ceiling ([cfd7bd1](https://github.com/AmbiqAI/helia-core-tester/commit/cfd7bd17e88c2cc99ec2252a0b6a2c139d1c157c))
* **hardware:** size 3x3 depthwise entry scratch for wide inputs ([1a92e78](https://github.com/AmbiqAI/helia-core-tester/commit/1a92e78cd3a0c8557b804df2732e78900512502e))
* **hardware:** size 3x3 depthwise entry scratch for wide inputs ([2e0ec1b](https://github.com/AmbiqAI/helia-core-tester/commit/2e0ec1b47b0eda57bb31f4cd17c0f7c7e336b880))
* **hardware:** skip nested repo state, time out git, narrow keys ([7b7e35c](https://github.com/AmbiqAI/helia-core-tester/commit/7b7e35cc6ec30416809d35f8f1c3e0e0af25063e))
* **hardware:** stream-only flag check ignores a moved checkout ([b949c17](https://github.com/AmbiqAI/helia-core-tester/commit/b949c17a4b5ac623dbe7273fb173bd75ec37fdac))
* **hardware:** strip comments and decode escapes before asm rules ([9782bd8](https://github.com/AmbiqAI/helia-core-tester/commit/9782bd88d7f8ddf889abe02a2cc1a10af09fbe5a))
* **hardware:** tie object scan to the tree, catch hidden addresses ([33fa032](https://github.com/AmbiqAI/helia-core-tester/commit/33fa032e5a78672e6be6755c0ff693d5f9dd9dfd))
* **hardware:** tie score to a passed candidate check ([30d7d53](https://github.com/AmbiqAI/helia-core-tester/commit/30d7d5326f8ddebf91becf378c6bca8cdcfac76f))
* **hardware:** tie score to a passed candidate check ([ba45ee1](https://github.com/AmbiqAI/helia-core-tester/commit/ba45ee10e33d165b6462b9e4e419f14214d3a5f3))
* **hardware:** trust kept mtimes only from fresh vendors ([31ff13b](https://github.com/AmbiqAI/helia-core-tester/commit/31ff13be3a4b6f60d2a1d1c1defe5a6ff0f2aa8a))
* **hardware:** trust old mtimes only for vendored bytes ([ecd2568](https://github.com/AmbiqAI/helia-core-tester/commit/ecd25683c88729ed1e88970c49ed82e6d065a700))
* **hardware:** unlink a linked module dir, resolve &lt;&gt; includes in Include/ ([aedfa1f](https://github.com/AmbiqAI/helia-core-tester/commit/aedfa1fecbde64ca78546461bce9fd2381ff54de))
* **hardware:** use pathutil containment in candidate check ([acaa805](https://github.com/AmbiqAI/helia-core-tester/commit/acaa80591e8f035c7c7437970ec06cb7208a651a))
* **hardware:** validate baseline.json, finite min-score, hidden missing cases ([96adc9c](https://github.com/AmbiqAI/helia-core-tester/commit/96adc9c3acd778601e854b34f970ff179d37cd01))
* **hardware:** vendor kernels into a fresh dir and swap it in ([b96c075](https://github.com/AmbiqAI/helia-core-tester/commit/b96c0750f58ddaebb2e8ab575d561bc5f00d61df))
* **hardware:** verify download TLS with certifi bundle ([1dc0c9b](https://github.com/AmbiqAI/helia-core-tester/commit/1dc0c9bf191094a595159c10f460a106b955d072))
* **hardware:** verify download TLS with certifi bundle ([9ca1206](https://github.com/AmbiqAI/helia-core-tester/commit/9ca12069812182125dede2afc8a9f048c4800b43))
* **hardware:** version scoring files, refuse old schemas ([d0cd23b](https://github.com/AmbiqAI/helia-core-tester/commit/d0cd23b86c2f74ba06048d8af43986f1c80ef42c))
* **mutation:** honor --ops, --workdir and --cmsis-nn-root in run ([4edf65b](https://github.com/AmbiqAI/helia-core-tester/commit/4edf65b5b41a0a38df924f8c4160fd7af7055ebd))
* **mutation:** honor --ops, --workdir and --cmsis-nn-root in run ([fe74a8d](https://github.com/AmbiqAI/helia-core-tester/commit/fe74a8d83444d8ec1ecf3567c0f0ef175a42d73f)), closes [#373](https://github.com/AmbiqAI/helia-core-tester/issues/373)
* **mutation:** size m55 cases' scratch for the host DSP build ([328c92f](https://github.com/AmbiqAI/helia-core-tester/commit/328c92f9685f00871f18539348ceaef7dc0eb33c))
* **mutation:** size m55 cases' scratch for the host DSP build ([d345142](https://github.com/AmbiqAI/helia-core-tester/commit/d3451429dbb781b702e00ef964bae5c99f2262db))
* **random-shapes:** validate every op token, tag cleanup from registry ([f7b3e22](https://github.com/AmbiqAI/helia-core-tester/commit/f7b3e22d168726a851bff6190cf402562d90a986))
* **score:** skip prepare growth gates on untouched cases ([c2e3ea2](https://github.com/AmbiqAI/helia-core-tester/commit/c2e3ea2830d66d1d0fa3a652486cce8d5d1190a7))
* **score:** skip prepare growth gates on untouched cases ([b242879](https://github.com/AmbiqAI/helia-core-tester/commit/b2428796166428c43632ba87f05a0d8a6689fcd6))
* **scripts:** fail ab_bundles when B breaks or drops cases ([0786040](https://github.com/AmbiqAI/helia-core-tester/commit/078604075c2807df034e3e3fbb1ec11564cbfd37))
* **scripts:** fail ab_bundles when B breaks or drops cases ([0290088](https://github.com/AmbiqAI/helia-core-tester/commit/0290088f5013f4b1c6e112f259dd36a7dc62191d)), closes [#374](https://github.com/AmbiqAI/helia-core-tester/issues/374)


### Performance

* **hardware:** build dir per placement, skip useless resets ([f59fb43](https://github.com/AmbiqAI/helia-core-tester/commit/f59fb43235e7ae0a4d3360c1a776608a6b2c83b4))
* **hardware:** build dir per placement, skip useless resets ([fd7f908](https://github.com/AmbiqAI/helia-core-tester/commit/fd7f908424b385ff6d1ae2e79376a4534083081a))
* **hardware:** cache gcc -E scans, check objects during stream ([6a168b0](https://github.com/AmbiqAI/helia-core-tester/commit/6a168b0202d42457c4e441c2ea5815119469ea5b))
* **hardware:** cache gcc -E scans, check objects during stream ([a95b16e](https://github.com/AmbiqAI/helia-core-tester/commit/a95b16e571d50cdbbe86ead27b76cc7f4f338f83))
* **hardware:** run every batch over one RTT session ([f841724](https://github.com/AmbiqAI/helia-core-tester/commit/f841724458e24e419c65dbb7d9b76c44d7903384))
* **hardware:** run every batch over one RTT session ([c4805cf](https://github.com/AmbiqAI/helia-core-tester/commit/c4805cf21eba327d568f16674b7cbbafccf6147a))
* **hardware:** stream blobs in 448-byte chunks ([c87788d](https://github.com/AmbiqAI/helia-core-tester/commit/c87788d887c6d59145bb64e42f004b14c3ef858a))
* **hardware:** stream blobs in 448-byte chunks ([ed3b9cc](https://github.com/AmbiqAI/helia-core-tester/commit/ed3b9cc88c687597218c8980460bc86c97faca05))
* **hardware:** write a wrapped RTT frame in one put ([a2bcbea](https://github.com/AmbiqAI/helia-core-tester/commit/a2bcbea48f7e4c98da127aee521fccc08e76bcd0))


### Refactoring

* **generation:** carry the 3x3 entry sizer on its entry row ([a49bcc0](https://github.com/AmbiqAI/helia-core-tester/commit/a49bcc079cd78b8f4e6292dda4ec234cd2fcee2b))
* **generation:** name the MLPerf dedupe key layer_key ([384258d](https://github.com/AmbiqAI/helia-core-tester/commit/384258d4a2997460450268a164a1d94653bc2e8a))
* **hardware:** drop unread golden manifest fields ([283967f](https://github.com/AmbiqAI/helia-core-tester/commit/283967f8f281b12c269e61ab9ee9cbcedfce5902))
* **hardware:** key packed FC scratch on its entry ([f22e5f6](https://github.com/AmbiqAI/helia-core-tester/commit/f22e5f6cbfd59ada008f71d265d4aee75d079878))
* **hardware:** name the field input_digest ([d11b262](https://github.com/AmbiqAI/helia-core-tester/commit/d11b262eec18187aa3eb509b045a44cbe1cd6787))
* **hardware:** one flag for flashed-build options ([b20a623](https://github.com/AmbiqAI/helia-core-tester/commit/b20a6230f493f83692a7171689c7b3eb8d1d34b9))
* **hardware:** reuse the typed extractor for tconv bias ([43891db](https://github.com/AmbiqAI/helia-core-tester/commit/43891db310da3896b1a99cc429d91e8c09fbb69e))
* **hardware:** tidy prepare gate docs and types ([4659c5d](https://github.com/AmbiqAI/helia-core-tester/commit/4659c5d0d2244385d7164bd2d46a991ff93c986b))
* **mutation:** resolve run paths at the option layer ([467e30d](https://github.com/AmbiqAI/helia-core-tester/commit/467e30d8eaa34a4fa588ce6513ff4aa869efe327))
* **random-shapes:** key generators by op, match descriptor stems ([ee2be7f](https://github.com/AmbiqAI/helia-core-tester/commit/ee2be7f12594b9d6ed38b70ddcac501b76f95f45))
* **random-shapes:** share the s16 tap limit and generic tail ([85b1ff3](https://github.com/AmbiqAI/helia-core-tester/commit/85b1ff3a1add5ee225a554584f1917435827f4b4))
* **random-shapes:** table the tc16 edge draws ([e6f6e56](https://github.com/AmbiqAI/helia-core-tester/commit/e6f6e56a581d48f1cdd5c3eb7aa4d01eb4396ee2))
* **scripts:** fold ab_bundles regressions into compare loop ([9a1de11](https://github.com/AmbiqAI/helia-core-tester/commit/9a1de11c765212dcaff675b384ecf08144256710))


### Docs

* agent loop flow, trust boundary and --skip-generate ([bbb7a10](https://github.com/AmbiqAI/helia-core-tester/commit/bbb7a109677f72d2d21a54da7380d033e0e3316c))
* agent loop flow, trust boundary and --skip-generate ([5f2aa06](https://github.com/AmbiqAI/helia-core-tester/commit/5f2aa0697f6d8c9fb2486f5e86a7dc079644b629))
* fix agent loop review findings ([cb97666](https://github.com/AmbiqAI/helia-core-tester/commit/cb97666fdb2e52f84a1b19983f54f0ab1682bca7))
* **hardware:** fix _tree_macros docstring ([beee182](https://github.com/AmbiqAI/helia-core-tester/commit/beee1825b34c2525024bb5c161ca82a29697a6b1))
* **hardware:** say tester bugs always print a traceback ([365c856](https://github.com/AmbiqAI/helia-core-tester/commit/365c8563440a09b305c73069f363340a3b467994))
* name the objects stage ([b557463](https://github.com/AmbiqAI/helia-core-tester/commit/b557463c54d0d238553f4a52b90939b9f5bf702b))
* note the 1xN forwarder off MVE ([b4db496](https://github.com/AmbiqAI/helia-core-tester/commit/b4db49628cdbf88f521654d864fc90a6904b9eb0))
* runnable score and check examples, copy limits ([e85dd70](https://github.com/AmbiqAI/helia-core-tester/commit/e85dd7020a5c9610d31ecd90e84f4bf4258116ce))

## [0.9.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.8.0...v0.9.0) (2026-10-04)


### Features

* **conv:** call direct entries with arm_convolve_1x1_s8_fast's signature ([#340](https://github.com/AmbiqAI/helia-core-tester/issues/340)) ([2591342](https://github.com/AmbiqAI/helia-core-tester/commit/2591342b90c41b8fe666326e28b2e81b2dbb50ce)), closes [#339](https://github.com/AmbiqAI/helia-core-tester/issues/339)
* **fc:** call arm_fully_connected_per_channel_packed_s8 as a direct entry ([#342](https://github.com/AmbiqAI/helia-core-tester/issues/342)) ([35704db](https://github.com/AmbiqAI/helia-core-tester/commit/35704db7153ee27e78fadbf23dc8759067d00eba)), closes [#341](https://github.com/AmbiqAI/helia-core-tester/issues/341)
* **hardware:** select run cases by op, dtype and case id ([596778a](https://github.com/AmbiqAI/helia-core-tester/commit/596778a5f76812f0afcee9f133e65d3f18bb3cca))


### Bug Fixes

* **hardware:** refuse --limit with --case-id ([9de7215](https://github.com/AmbiqAI/helia-core-tester/commit/9de7215a3050b9d7cb1d08a3e11609586d0d242c))
* **hardware:** refuse an empty --cases-from file ([5f0531b](https://github.com/AmbiqAI/helia-core-tester/commit/5f0531b4f245d59ff9b9404fe04ec8fcecc87c19))

## [0.8.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.7.0...v0.8.0) (2026-10-02)


### Features

* **hardware:** add apollo330mP_evb as a second board ([428c59b](https://github.com/AmbiqAI/helia-core-tester/commit/428c59bbebb0b9a0c6c5526983714304a8bbb621))
* **hardware:** add apollo3p_evb Cortex-M4 DWT-only board ([6ad9561](https://github.com/AmbiqAI/helia-core-tester/commit/6ad95618de0a0a94f1a0c8faf6327da32981cfb2))
* **hardware:** board matrix runner and per-board run paths ([afdf9ef](https://github.com/AmbiqAI/helia-core-tester/commit/afdf9ef1ff039202aafbf4c800c8b709131daa38))
* **hardware:** name the case and target state on a stall ([2575612](https://github.com/AmbiqAI/helia-core-tester/commit/25756129d81117628919914a81a7bdd2f4627c7f))
* **hardware:** refuse boards booted at the wrong clock ([8b3a11f](https://github.com/AmbiqAI/helia-core-tester/commit/8b3a11f150aa4111ebc9c6369f654e93ee54a66b))
* **hardware:** refuse boards booted at the wrong clock ([35be217](https://github.com/AmbiqAI/helia-core-tester/commit/35be217dc232f2c7afec7e69cb0c493c7649d0fb))
* **hardware:** report boot health in TARGET_INFO ([3e85336](https://github.com/AmbiqAI/helia-core-tester/commit/3e8533645fba2628c50250e0f654310ace3cb69b))
* **hardware:** version the nightly run document and record build sources ([75ef4e8](https://github.com/AmbiqAI/helia-core-tester/commit/75ef4e8ba6498fb5d5cab5e7498885cadc82df51))
* **hardware:** version the nightly run document and record build sources ([626db66](https://github.com/AmbiqAI/helia-core-tester/commit/626db663012270edf16bef4852098c1125aea05f))
* **scripts:** add board matrix runner and summary ([a7d71e5](https://github.com/AmbiqAI/helia-core-tester/commit/a7d71e562154fd0fdf60ab925587e2a4243e8cc2))


### Bug Fixes

* **arg:** float ARG_MAX/ARG_MIN expectations never let a NaN win ([#318](https://github.com/AmbiqAI/helia-core-tester/issues/318)) ([91581e9](https://github.com/AmbiqAI/helia-core-tester/commit/91581e954eaa67a7387aa6d67ff455e4c9437cec)), closes [#310](https://github.com/AmbiqAI/helia-core-tester/issues/310)
* **hardware:** classify memory report sections by region address ([47b58fe](https://github.com/AmbiqAI/helia-core-tester/commit/47b58fe55ce4627f71a13c688e4bc5ab119a5336))
* **hardware:** classify memory report sections by region address ([cd78dd2](https://github.com/AmbiqAI/helia-core-tester/commit/cd78dd2476ac47e476697469e65e701c50be3bc0))
* **hardware:** drain RTT rings by direct memory access ([f7f09c1](https://github.com/AmbiqAI/helia-core-tester/commit/f7f09c19e74b9d57b314229a39792b09c31c3758))
* **hardware:** drain RTT rings by direct memory access ([e764d5a](https://github.com/AmbiqAI/helia-core-tester/commit/e764d5af655e2fec8013800abc17f47dc76b4295))
* **hardware:** fail the link on a short RTT ring access ([9028f69](https://github.com/AmbiqAI/helia-core-tester/commit/9028f69eda94cf5d120de556f85a02ff5ef39b40))
* **hardware:** key per-run paths by board for concurrent runs ([0dbc3ca](https://github.com/AmbiqAI/helia-core-tester/commit/0dbc3caa3f2f87a9a50cb5ce86f9effe994db475))
* **hardware:** record the GCC CMake configured, not the PATH one ([f49fd67](https://github.com/AmbiqAI/helia-core-tester/commit/f49fd670d39c097a738dc76e50c89ec98aac81b2))
* **hardware:** record the precision and FVP gate the run used ([72bbd96](https://github.com/AmbiqAI/helia-core-tester/commit/72bbd96eca35b0ba533ba6473dc525178c1187cd))
* **hardware:** record the precision and FVP gate the run used ([3458529](https://github.com/AmbiqAI/helia-core-tester/commit/3458529833bd9a059a29880ed14bf485d9dfdedd))
* **mutation:** re-anchor the 1xN s8 guard mutants ([689c2dc](https://github.com/AmbiqAI/helia-core-tester/commit/689c2dc9a31be9de79c75d130230f2eeab73833e))
* **scripts:** forward --precision from the board matrix ([f11990c](https://github.com/AmbiqAI/helia-core-tester/commit/f11990c56dbcba16a89de01f4cb9c33b0e4fb74e))
* **scripts:** refuse matrix-owned args and broken bundles ([1d05c18](https://github.com/AmbiqAI/helia-core-tester/commit/1d05c1894e5a15574a09a0376a96ea2dfe8f5172))
* **scripts:** refuse non-finite values, keep errored boards as null ([5dd344f](https://github.com/AmbiqAI/helia-core-tester/commit/5dd344fbc909a861222a2deae4dc7876f6f8bdbd))
* **scripts:** refuse shared build dir, treat any bad read as bad bundle ([dec06a5](https://github.com/AmbiqAI/helia-core-tester/commit/dec06a5fe5629dcb19d54087760f050983f93f4e))
* **scripts:** stabilize board matrix rows and probe env ([834a34f](https://github.com/AmbiqAI/helia-core-tester/commit/834a34f26787ca75754883055ff8f24f0dabcc7f))
* **scripts:** type-check bundle counts, survive launch errors ([978a419](https://github.com/AmbiqAI/helia-core-tester/commit/978a419ef29828e3115b54f9bde960cd7867ab12))


### Refactoring

* **hardware:** require run options and record narrowing flags ([e634401](https://github.com/AmbiqAI/helia-core-tester/commit/e634401356b16a665f9adec304c52764f9aed9a3))
* **hardware:** share the MHz clock formatter ([c5f84dc](https://github.com/AmbiqAI/helia-core-tester/commit/c5f84dc001315fa5cf39b810b30c6bf1e12b1665))


### Docs

* describe the board clock check ([01ab3d4](https://github.com/AmbiqAI/helia-core-tester/commit/01ab3d40bb4c131c6b09d56b339d3f5b4359e7eb))

## [0.7.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.6.0...v0.7.0) (2026-10-01)


### Features

* **hardware:** drive the PMU through the nsx-pmu-armv8m module ([1f28b96](https://github.com/AmbiqAI/helia-core-tester/commit/1f28b96069d3b4fe5032b8430d680d944d7305eb))
* **hardware:** drive the PMU through the nsx-pmu-armv8m module ([9af1500](https://github.com/AmbiqAI/helia-core-tester/commit/9af15003a5ab5d95fcbcf21802720032f1d34e1e))
* **hardware:** flash and run what was built ([6e8fce9](https://github.com/AmbiqAI/helia-core-tester/commit/6e8fce9fa4087b48f914b9c50a8c27b50c7b2de5))
* **hardware:** keep the session going when a kernel rejects a case ([886e4a6](https://github.com/AmbiqAI/helia-core-tester/commit/886e4a662b28db2f9272425f9e1cf5390e828df6))
* **hardware:** keep the session going when a kernel rejects a case ([bc57107](https://github.com/AmbiqAI/helia-core-tester/commit/bc5710719fddb53d9fb917d8f6d39d3bcbe22fde))
* **hardware:** pin ns-cmsis-nn v7.38.0 and follow pin bumps ([5fd7655](https://github.com/AmbiqAI/helia-core-tester/commit/5fd76557f7cc242764d37b39fbc295f8c5ceaa14))
* **hardware:** pin ns-cmsis-nn v7.38.0 and follow pin bumps ([eb5aeb4](https://github.com/AmbiqAI/helia-core-tester/commit/eb5aeb4c96522f5917a02be3969a7056eaf171f4))
* **hardware:** run the full PMU catalog in one session ([588854f](https://github.com/AmbiqAI/helia-core-tester/commit/588854ff7e70a96304e974f2a00bd7b87470ae81))
* **hardware:** run the full PMU catalog in one session ([94cae1a](https://github.com/AmbiqAI/helia-core-tester/commit/94cae1aa04a0fae7e0c7ec5e27991b5ec980f27c))
* **hardware:** stamp bundles with NSX build provenance ([52809cf](https://github.com/AmbiqAI/helia-core-tester/commit/52809cf67df26c92008f7640be6fbd84ece06c49))
* **hardware:** stamp bundles with NSX build provenance ([8ea745a](https://github.com/AmbiqAI/helia-core-tester/commit/8ea745afe6e0be2fb4f17eab85f28093edaa0df2))
* **pmu:** sync the PMU catalog from nsx-pmu-armv8m ([4119e69](https://github.com/AmbiqAI/helia-core-tester/commit/4119e6927bc5e65e10fcb4d5fe4a8babad5d32d9))
* **pmu:** sync the PMU catalog from nsx-pmu-armv8m ([7c59985](https://github.com/AmbiqAI/helia-core-tester/commit/7c59985b6ee80a9811ca51a53b01b93dc49e205b))


### Bug Fixes

* **hardware:** fake target rejects unmapped PMU event ids ([e024d09](https://github.com/AmbiqAI/helia-core-tester/commit/e024d09447d5f77b22ea9a905c76e35f715d426f))
* **hardware:** fake target rejects unmapped PMU event ids ([c358c55](https://github.com/AmbiqAI/helia-core-tester/commit/c358c5570c51a5b728a245268ceeff585a311b9d))
* **hardware:** let the float suite run end to end on hardware ([d24435c](https://github.com/AmbiqAI/helia-core-tester/commit/d24435ca8070b2f8d7cf9444b76e14e4aeb6254e))
* **hardware:** null unverified or corrupt bundle provenance ([ccea531](https://github.com/AmbiqAI/helia-core-tester/commit/ccea531d85aa29d28faef7ea534ff49e7f205442))
* **hardware:** record NSX version and checkout at build time ([d1732fb](https://github.com/AmbiqAI/helia-core-tester/commit/d1732fb129331abb6ad893b510c77db18125c6cc))
* **hardware:** refuse bare all mixed with groups ([0f7fcda](https://github.com/AmbiqAI/helia-core-tester/commit/0f7fcda79bc4542821fa0d6c5f39be7201c76bd0))
* **hardware:** refuse results that overflow the firmware outbox ([fb46a10](https://github.com/AmbiqAI/helia-core-tester/commit/fb46a10499888e04ffd6a95e34974f9a5baec9b7))
* **hardware:** render the size probe from the requested checkout ([be75b41](https://github.com/AmbiqAI/helia-core-tester/commit/be75b4157047fc3e43329e18cad5eee0b0865246))
* **hardware:** run benchmark firmware at 250 MHz ([a0b8dc2](https://github.com/AmbiqAI/helia-core-tester/commit/a0b8dc2ad0199541cc45c379511556424c9ddcf5))
* **hardware:** run benchmark firmware at 250 MHz ([e9881f1](https://github.com/AmbiqAI/helia-core-tester/commit/e9881f1c538a5c3b00cdbd558f52cc3cae0ca470))
* **hardware:** send padded packed float conv weights ([04301f6](https://github.com/AmbiqAI/helia-core-tester/commit/04301f657dcd07504716aa286926cca9a37160be))
* **hardware:** skip float cases the firmware rejects ([ce38fcf](https://github.com/AmbiqAI/helia-core-tester/commit/ce38fcf6c6ff894e5fc01fb7fa98106a8c204717))
* **hardware:** stage float stream cases under float/ ([ee3b558](https://github.com/AmbiqAI/helia-core-tester/commit/ee3b55889b9e1b32f9117aa02d1c7f84ade8be91))
* **hardware:** tighten the outbox check after review ([754186f](https://github.com/AmbiqAI/helia-core-tester/commit/754186f589705218d3222867fecca25dbc5f7da7))
* **hardware:** time the kernel itself, not its trampoline ([b2e49ff](https://github.com/AmbiqAI/helia-core-tester/commit/b2e49ff39fa9a7aa5bc40f522430e119471260a5))
* **hardware:** wait one read timeout per PMU pass while sampling ([b525f01](https://github.com/AmbiqAI/helia-core-tester/commit/b525f01d0d2bd713a5c633f9c1ed00ff5b414953))
* **mutation:** re-anchor three mutants on ns-cmsis-nn main ([3980a3e](https://github.com/AmbiqAI/helia-core-tester/commit/3980a3ef36afe952a3ee7adc8d9f224c5f6c8416))
* **mutation:** re-anchor three mutants on ns-cmsis-nn main ([7d5c439](https://github.com/AmbiqAI/helia-core-tester/commit/7d5c439c25b79b1bdb32e94eef504717254bae4d))
* **pmu:** compare catalog bytes; relock before re-sync ([8961f60](https://github.com/AmbiqAI/helia-core-tester/commit/8961f6044ee13d4ba75f092fe377aab8f5ef3637))


### Performance

* **hardware:** count only kernel calls in timed samples ([cbf1d39](https://github.com/AmbiqAI/helia-core-tester/commit/cbf1d3921d4f60cdc22856bb0b6d69aa75274bc2))
* **hardware:** count only kernel calls in timed samples ([c80ea2e](https://github.com/AmbiqAI/helia-core-tester/commit/c80ea2e9c9719ccf9ac08090d3aefdc720dfdc2c))


### Refactoring

* **hardware:** drop the CMake hardware path ([31463a9](https://github.com/AmbiqAI/helia-core-tester/commit/31463a952d2fd4f0cef27032e01b4ded3ecdfc84))
* **hardware:** keep explicit refs from older records ([a688921](https://github.com/AmbiqAI/helia-core-tester/commit/a6889210f0feeb9faf49b349a459fd1a2900436e))
* **hardware:** keep timed-call helpers clear of kernel-contracts ([9ea7696](https://github.com/AmbiqAI/helia-core-tester/commit/9ea769650e4b18a20758b16483689a2cad8052f3))
* **hardware:** tighten float bridge guards after review ([d51688d](https://github.com/AmbiqAI/helia-core-tester/commit/d51688d4123c31793a50da8156650d5a0e04cc25))
* **hardware:** tighten rejection handling after review ([7770e5c](https://github.com/AmbiqAI/helia-core-tester/commit/7770e5c72997148c497aedb6f4bfd15c00249882))


### Docs

* drop claims that CI pytest has no ns-cmsis-nn ([0ce3586](https://github.com/AmbiqAI/helia-core-tester/commit/0ce3586437fa855b23de0cacacebf9e27b9beec8))
* **hardware:** say PMU firmware rejects unmapped event ids ([7c7bbaa](https://github.com/AmbiqAI/helia-core-tester/commit/7c7bbaa92322edeaecc816f8a3fd9a600ded6b5e))

## [0.6.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.5.0...v0.6.0) (2026-09-29)


### Features

* **hardware:** build the firmware through neuralspotx ([10fbcb5](https://github.com/AmbiqAI/helia-core-tester/commit/10fbcb59a871034a2dcb2b78cfbc70bac5813b14))


### Bug Fixes

* **conv:** size float scratch for long patches and cover long FP16 reductions ([#249](https://github.com/AmbiqAI/helia-core-tester/issues/249)) ([de3a89a](https://github.com/AmbiqAI/helia-core-tester/commit/de3a89a2738431e1b280bed54c5b93c01094b4ae)), closes [#248](https://github.com/AmbiqAI/helia-core-tester/issues/248)
* **hardware:** drop dead RTT channel 0 buffers ([3028c07](https://github.com/AmbiqAI/helia-core-tester/commit/3028c07dd97795c5ac777bc824c0bfb9bdcf5de8))
* **hardware:** drop dead RTT channel 0 buffers ([fb1a939](https://github.com/AmbiqAI/helia-core-tester/commit/fb1a939a8df0475d24a32b05b639c7e5728c8e66))

## [0.5.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.4.0...v0.5.0) (2026-09-29)


### Features

* **conv:** call the int8 convolve direct entries from a descriptor ([#243](https://github.com/AmbiqAI/helia-core-tester/issues/243)) ([68adad5](https://github.com/AmbiqAI/helia-core-tester/commit/68adad587049a74be7178da253a86f3e200782d6)), closes [#241](https://github.com/AmbiqAI/helia-core-tester/issues/241)
* **entries:** call the FP16 _acc16 and layout-free entries from a descriptor ([#247](https://github.com/AmbiqAI/helia-core-tester/issues/247)) ([fa4b379](https://github.com/AmbiqAI/helia-core-tester/commit/fa4b379b230fbeddbe1d580f41f29f9c1c597f4b)), closes [#242](https://github.com/AmbiqAI/helia-core-tester/issues/242)

## [0.4.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.3.3...v0.4.0) (2026-09-28)


### Features

* **coverage:** add a cortex-m55 coverage lane for integer MVE paths ([#234](https://github.com/AmbiqAI/helia-core-tester/issues/234)) ([fb57a0d](https://github.com/AmbiqAI/helia-core-tester/commit/fb57a0dd393cbe0f81f2e722c9ad2c64851d939f)), closes [#233](https://github.com/AmbiqAI/helia-core-tester/issues/233)
* **depthwise:** call a named ns-cmsis-nn depthwise entry from a descriptor ([#238](https://github.com/AmbiqAI/helia-core-tester/issues/238)) ([4af62bf](https://github.com/AmbiqAI/helia-core-tester/commit/4af62bf99df3c676bcebacf514812cb4913aaba5)), closes [#237](https://github.com/AmbiqAI/helia-core-tester/issues/237)

## [0.3.3](https://github.com/AmbiqAI/helia-core-tester/compare/v0.3.2...v0.3.3) (2026-09-28)


### Tests

* **conv:** cover pending ns-cmsis-nn conv, pointwise and 1xk depthwise paths ([#230](https://github.com/AmbiqAI/helia-core-tester/issues/230)) ([989c73a](https://github.com/AmbiqAI/helia-core-tester/commit/989c73a7b51539d34fd6704fb005fe084ef6288b))

## [0.3.2](https://github.com/AmbiqAI/helia-core-tester/compare/v0.3.1...v0.3.2) (2026-09-28)


### Tests

* **depthwise:** cover generic s4 depthwise with odd channel counts and padding ([#223](https://github.com/AmbiqAI/helia-core-tester/issues/223)) ([1e6648c](https://github.com/AmbiqAI/helia-core-tester/commit/1e6648ca0d898fa12aa2babe34df16fdb39e1f1b))
* **depthwise:** cover the s4 optimized depthwise channel and pixel tails ([#226](https://github.com/AmbiqAI/helia-core-tester/issues/226)) ([844e927](https://github.com/AmbiqAI/helia-core-tester/commit/844e9274fe7114c29f73fc63b3e51cee64132b2b))

## [0.3.1](https://github.com/AmbiqAI/helia-core-tester/compare/v0.3.0...v0.3.1) (2026-09-27)


### Bug Fixes

* **conv:** compute convolve and transpose conv per-channel scales in double ([#219](https://github.com/AmbiqAI/helia-core-tester/issues/219)) ([9f865b0](https://github.com/AmbiqAI/helia-core-tester/commit/9f865b0b1a4b7c85af89c5cb6f4165bc030c72fe)), closes [#215](https://github.com/AmbiqAI/helia-core-tester/issues/215)

## [0.3.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.2.0...v0.3.0) (2026-09-26)


### Features

* add arm_gru_unidirectional_f16 test generation support ([4839869](https://github.com/AmbiqAI/helia-core-tester/commit/483986940bf8658bb65eef2a3a896bba0cc5abee))
* add comparison parameters for tanh activation functions in float tests ([dcd0b6c](https://github.com/AmbiqAI/helia-core-tester/commit/dcd0b6c0ef61a6cbd1f3f025b2b93fc0484880f8))
* add float ReduceSum coverage (arm_reduce_sum_f32/f16) ([a43a4b2](https://github.com/AmbiqAI/helia-core-tester/commit/a43a4b2ce70fca64fe57a93b7d0bb6d1094945c7))
* Add FP16 test coverage for arm_split_f16, arm_strided_slice_f16, arm_elementwise_sub_f16 ([82dfd41](https://github.com/AmbiqAI/helia-core-tester/commit/82dfd413f806c37783d627eb8025cc4734df1aa4))
* add FP32/FP16 coverage for abs, sub, prelu, and strided_slice ([709b70e](https://github.com/AmbiqAI/helia-core-tester/commit/709b70e7df9c595849375c6f008ad97ee708b9b8))
* add new BroadcastTo and DynamicUpdateSlice operations with error handling ([e353d83](https://github.com/AmbiqAI/helia-core-tester/commit/e353d83d44fbeae5dc29851a7815046f1045031d))
* add new float operations and coverage cleanup functionality ([9173ca7](https://github.com/AmbiqAI/helia-core-tester/commit/9173ca7a03afcfd5348c28d539a3784ee4dc53d1))
* Add sqrt tests and rsqrt support ([3e5da7d](https://github.com/AmbiqAI/helia-core-tester/commit/3e5da7dd4d4313272a504160134970ae2935e4df))
* add tensor shaping operator tests (tile, broadcast_to, scatter_nd, mirror_pad, select_v2, where, reverse_sequence, dynamic_update_slice) ([67b9d1c](https://github.com/AmbiqAI/helia-core-tester/commit/67b9d1c791a1f7befaf0306a1a972379025675e5))
* Add zero-size reshape cases and new transpose cases for FP16 and FP32 ([a2a4d02](https://github.com/AmbiqAI/helia-core-tester/commit/a2a4d026d227568537de26468aed9013196c35ec))
* **arg:** add standalone float extrema fixtures ([f87cc90](https://github.com/AmbiqAI/helia-core-tester/commit/f87cc9098b7ef7e0d3628f43a30a5852068982d5))
* call the ns-cmsis-nn LSTM/GRU temp-buffer sizers when the checkout declares them ([#90](https://github.com/AmbiqAI/helia-core-tester/issues/90)) ([d0b0f92](https://github.com/AmbiqAI/helia-core-tester/commit/d0b0f92beaf4d977654c8c4c52aabd5edc8bd32d))
* chunked-equivalence cases for the elementwise families ([#88](https://github.com/AmbiqAI/helia-core-tester/issues/88)) ([ca637d9](https://github.com/AmbiqAI/helia-core-tester/commit/ca637d942a0cbbc5dfff084b5b7d412a28c8548e))
* **ci:** add self-validating CI against ns-cmsis-nn ([#61](https://github.com/AmbiqAI/helia-core-tester/issues/61)) ([862e991](https://github.com/AmbiqAI/helia-core-tester/commit/862e9912683e7afcf6e8df45c9e2b6d29f243627))
* **cli:** --pmu-counters GROUP:SELECTION, per-counter bundle columns and stage timing ([24a7486](https://github.com/AmbiqAI/helia-core-tester/commit/24a748616a248b111b8ddba6bcfe35b38e932a8f))
* **cli:** replace perf-stream group and run_hardware_perf_suite.sh with a board-keyed hardware CLI ([aef72c0](https://github.com/AmbiqAI/helia-core-tester/commit/aef72c04b0e0f82106dd437ad8a6b99cc1674876))
* **coverage:** merge cortex-m55 MVE float coverage ([29a9b95](https://github.com/AmbiqAI/helia-core-tester/commit/29a9b95ce081ded54a8e4d0c3a028a97bafad7ab))
* **descriptors:** cover float sqrt and reciprocal sqrt ([#129](https://github.com/AmbiqAI/helia-core-tester/issues/129)) ([199bbeb](https://github.com/AmbiqAI/helia-core-tester/commit/199bbeb53c5fb57211211ebf748bff6536d0cea9))
* **descriptors:** float copy-class kernels for ns-cmsis-nn[#475](https://github.com/AmbiqAI/helia-core-tester/issues/475) ([#128](https://github.com/AmbiqAI/helia-core-tester/issues/128)) ([b82b2a2](https://github.com/AmbiqAI/helia-core-tester/commit/b82b2a2f6e3e32dea7a1a8d732775d6988216e45))
* **descriptors:** float elementwise broadcast shapes and a dims-taking float call path (ns-cmsis-nn[#415](https://github.com/AmbiqAI/helia-core-tester/issues/415)) ([#111](https://github.com/AmbiqAI/helia-core-tester/issues/111)) ([fa98ca9](https://github.com/AmbiqAI/helia-core-tester/commit/fa98ca97cef2ae3cd52b3ec48b3405f512f9dec3))
* **descriptors:** float16 squared difference cases with full kernel coverage (ns-cmsis-nn[#490](https://github.com/AmbiqAI/helia-core-tester/issues/490)) ([#189](https://github.com/AmbiqAI/helia-core-tester/issues/189)) ([05c83bb](https://github.com/AmbiqAI/helia-core-tester/commit/05c83bb4d798e0fcfc12cf244011496646cafbf2))
* **descriptors:** hard_swish s8 sizes that reach the table path (ns-cmsis-nn[#289](https://github.com/AmbiqAI/helia-core-tester/issues/289)) ([#118](https://github.com/AmbiqAI/helia-core-tester/issues/118)) ([4e0e6e5](https://github.com/AmbiqAI/helia-core-tester/commit/4e0e6e59dbd0a49b3ba5f166930970087b30f4c4))
* **descriptors:** small-input-channel float conv and large-K f16 FC shapes (ns-cmsis-nn[#417](https://github.com/AmbiqAI/helia-core-tester/issues/417)) ([#101](https://github.com/AmbiqAI/helia-core-tester/issues/101)) ([28f1a8a](https://github.com/AmbiqAI/helia-core-tester/commit/28f1a8a14a07d25de58ea13e6451248f9c6bde00))
* disable PReLU and StridedSlice test cases due to LiteRT invocation failures ([978d835](https://github.com/AmbiqAI/helia-core-tester/commit/978d83502ac155da006a1b5f4eba95a634343049))
* Enhance runtime environment management and coverage reporting ([5ce2166](https://github.com/AmbiqAI/helia-core-tester/commit/5ce2166efb0a2bda8cc79a42f91d52b7259d35d4))
* expand float test cases ([93215a6](https://github.com/AmbiqAI/helia-core-tester/commit/93215a68aa5f4d3aa09194cba59e5bd358ddb794))
* float mean and hard_swish operator support with per-symbol codegen probing ([#92](https://github.com/AmbiqAI/helia-core-tester/issues/92)) ([d91c3e8](https://github.com/AmbiqAI/helia-core-tester/commit/d91c3e8fc1f8142310c52eb6228aba27b3b27d98))
* float profile summary table ([1421b70](https://github.com/AmbiqAI/helia-core-tester/commit/1421b70627071bc8e2ee13a254faa99486051297))
* **generation:** block-size invariance for the remaining int kernels and an operand sign-span rule ([#115](https://github.com/AmbiqAI/helia-core-tester/issues/115)) ([cce1bd1](https://github.com/AmbiqAI/helia-core-tester/commit/cce1bd153f693314627b53c9df6e532839b59a63))
* **generation:** fault-injection cases for the non-recurrent operator families ([#126](https://github.com/AmbiqAI/helia-core-tester/issues/126)) ([fafd41a](https://github.com/AmbiqAI/helia-core-tester/commit/fafd41a9ef3bed718a952901b0bd19315446edba))
* **generation:** give int Convolve and FullyConnected cases a detectable bias ([#102](https://github.com/AmbiqAI/helia-core-tester/issues/102)) ([b44c2e6](https://github.com/AmbiqAI/helia-core-tester/commit/b44c2e66d630a74a6650955c0c1845bd3724e78e))
* **gru,lstm:** close coverage backlog from issue [#56](https://github.com/AmbiqAI/helia-core-tester/issues/56) ([#82](https://github.com/AmbiqAI/helia-core-tester/issues/82)) ([d8fa13f](https://github.com/AmbiqAI/helia-core-tester/commit/d8fa13f5ff664ea33267a931d1cd1ffe17084524))
* **gru:** add FP32 coverage for arm_gru_unidirectional_f32 ([ff77d21](https://github.com/AmbiqAI/helia-core-tester/commit/ff77d218a8319f43968085fcacd1f92b30b2a403))
* **hardware:** add A/B bundle comparison script ([9b006e4](https://github.com/AmbiqAI/helia-core-tester/commit/9b006e4c561cb9c58eac31090ec9fcbda948696e))
* **hardware:** add neuralspotx dependency and API facade ([b3e2b80](https://github.com/AmbiqAI/helia-core-tester/commit/b3e2b80dec6b11fda1ff3d87b65bc872af791f34))
* **hardware:** pin nsx-segger-rtt to v0.1.1 ([8e63b78](https://github.com/AmbiqAI/helia-core-tester/commit/8e63b784291b7cae5559af6df24373edb1fdfff5))
* **hardware:** render the firmware as an NSX app ([6361419](https://github.com/AmbiqAI/helia-core-tester/commit/6361419e1fbb70cacf4711f2aba3366fc2cec123))
* **harness:** guard bytes for buffer overruns in generated tests ([#95](https://github.com/AmbiqAI/helia-core-tester/issues/95)) ([22bde70](https://github.com/AmbiqAI/helia-core-tester/commit/22bde702675b6022d1ecce021c2897db12dd3ed4))
* **harness:** size SVDF ctx scratch with the published sizers and assert exact values ([#121](https://github.com/AmbiqAI/helia-core-tester/issues/121)) ([fd46398](https://github.com/AmbiqAI/helia-core-tester/commit/fd463989bc4e989e360b95cd9e06ca1d6d56c7cc))
* hw benchmark pmu ([#85](https://github.com/AmbiqAI/helia-core-tester/issues/85)) ([51ed108](https://github.com/AmbiqAI/helia-core-tester/commit/51ed108dc4b1bd954b8196e3b44164dea089189d))
* mutation scoring for the int elementwise and conv families ([#89](https://github.com/AmbiqAI/helia-core-tester/issues/89)) ([07f0b22](https://github.com/AmbiqAI/helia-core-tester/commit/07f0b22486f71111b45b7ffd332b57c2cc57c402))
* non-finite float inputs and a cortex-m0 f32 leg ([#97](https://github.com/AmbiqAI/helia-core-tester/issues/97)) ([4ab6e3e](https://github.com/AmbiqAI/helia-core-tester/commit/4ab6e3eec9b21b913bdeb47777572caaf51aee5f))
* **nonfinite:** assert the GRU non-finite cases strictly under the public NaN contract ([#125](https://github.com/AmbiqAI/helia-core-tester/issues/125)) ([af7ac9f](https://github.com/AmbiqAI/helia-core-tester/commit/af7ac9f531da0c5d8903ba5bbfe0ce0bf4c7ea2a))
* **nonfinite:** masked comparison policy and non-finite cover for the remaining pointwise, pass-through and reduction float families ([#103](https://github.com/AmbiqAI/helia-core-tester/issues/103)) ([4b5fcbe](https://github.com/AmbiqAI/helia-core-tester/commit/4b5fcbe36cc69c53ff2362435058cf8f4f199b97))
* **nonfinite:** non-finite cover for the conv family and batch matmul, seeded depthwise generation ([#110](https://github.com/AmbiqAI/helia-core-tester/issues/110)) ([ffaabe6](https://github.com/AmbiqAI/helia-core-tester/commit/ffaabe6f0435906471d1ed6dbc0afc16e130495e))
* **nonfinite:** non-finite input sweeps for the recurrent float families ([#108](https://github.com/AmbiqAI/helia-core-tester/issues/108)) ([397c984](https://github.com/AmbiqAI/helia-core-tester/commit/397c984b8ea4b0ba59a2d7c39e210360be90f73d))
* **perf-stream:** 32-case sessions batched by LOAD_PLAN size ([2a70108](https://github.com/AmbiqAI/helia-core-tester/commit/2a70108c5889ad8b0700a6a1aa8c7601859c0c82))
* **perf-stream:** add the board table and J-Link probe resolution ([5c49dd6](https://github.com/AmbiqAI/helia-core-tester/commit/5c49dd66ed5c9ee10b9ec0f116c016fba6cc17f9))
* **perf-stream:** chained PMU event-counter capture (HCTP v2, --pmu-counters, 32-case sessions) ([4350ad7](https://github.com/AmbiqAI/helia-core-tester/commit/4350ad77bf8360fbdf9624c8960b2ddf7eccaf61))
* **perf-stream:** hash the whole linked image for the firmware build id ([3373da1](https://github.com/AmbiqAI/helia-core-tester/commit/3373da1d1e5443213e18a27c4c4720c8e1918da8))
* **perf-stream:** HCTP v2 with chained PMU event-counter passes ([a9a24b7](https://github.com/AmbiqAI/helia-core-tester/commit/a9a24b745cdfac6db251d2a42d966c0e00611e17))
* **pipeline:** generation reuse stamp, bounded parallel default, per-case timeout default, compiler-cache hook ([#112](https://github.com/AmbiqAI/helia-core-tester/issues/112)) ([26a11d2](https://github.com/AmbiqAI/helia-core-tester/commit/26a11d294e3ccecc1b656028b70bb73c9a34a8f7))
* **prelu:** add s16 PReLU test coverage ([26a5230](https://github.com/AmbiqAI/helia-core-tester/commit/26a523030de85a8838263a6a1379411793b99ffd))
* **reduce:** add bit-exact float extrema descriptors ([082fbd6](https://github.com/AmbiqAI/helia-core-tester/commit/082fbd66ea00abbc40d158b541f7b1476e4d32a1))
* Refactor and enhance operations with new configurations and tests and add test cases for better coverage ([d953ed1](https://github.com/AmbiqAI/helia-core-tester/commit/d953ed1429dda52ab88c50fb97da17a87fe9be1a))
* Refactor tag release workflow to automate tagging on PR merges ([991f38a](https://github.com/AmbiqAI/helia-core-tester/commit/991f38a746a05401b7e7efbd9ffd962ded3dde03))
* **runtime:** emit measured max-diff/tolerance-fraction for float cases ([#78](https://github.com/AmbiqAI/helia-core-tester/issues/78)) ([a9c4233](https://github.com/AmbiqAI/helia-core-tester/commit/a9c4233feb19d68a249e6c837b18afa6692daa89))
* **sizer:** report a bad scratch-sizer answer as a sizer failure ([#133](https://github.com/AmbiqAI/helia-core-tester/issues/133)) ([#152](https://github.com/AmbiqAI/helia-core-tester/issues/152)) ([8dcca91](https://github.com/AmbiqAI/helia-core-tester/commit/8dcca91e57f247963661745d7702af26a1e850ed))
* **summary:** add profile rows for fallback and mve float reports ([816d366](https://github.com/AmbiqAI/helia-core-tester/commit/816d366b9378aad1c09e05dd34a5829ce8502b66))


### Bug Fixes

* **abs:** follow ns-cmsis-nn's arm_nn_abs_f16/f32 rename ([5b90642](https://github.com/AmbiqAI/helia-core-tester/commit/5b906424d1796ca19bdd66e34d1b591e9f73dd43))
* **activation:** align scalar FP16 tanh reference with LUT ([#187](https://github.com/AmbiqAI/helia-core-tester/issues/187)) ([a999cfa](https://github.com/AmbiqAI/helia-core-tester/commit/a999cfaba48b6a33f08f65e068db3e174d020869)), closes [#134](https://github.com/AmbiqAI/helia-core-tester/issues/134)
* add input_1/2_shape to select_v2 descriptor for validation ([84ec970](https://github.com/AmbiqAI/helia-core-tester/commit/84ec970a6cc723702a9258275bc8ffec2c171796))
* address PR review comments ([1234677](https://github.com/AmbiqAI/helia-core-tester/commit/123467705f74a763dba6e082ff18a8ede7f3220c))
* **benchmark:** fail the case when a benchmarked call fails ([#197](https://github.com/AmbiqAI/helia-core-tester/issues/197)) ([e428f18](https://github.com/AmbiqAI/helia-core-tester/commit/e428f18afba8bc46be3d4e199a76e50b0225369b)), closes [#146](https://github.com/AmbiqAI/helia-core-tester/issues/146)
* **bias:** gate nonzero bias to float cases; emit int guard only when used ([928c2f0](https://github.com/AmbiqAI/helia-core-tester/commit/928c2f025e3009d21a999d1d3e95d4aee86b4fe5))
* **bias:** give float Convolve/FullyConnected cases real bias data ([4b82314](https://github.com/AmbiqAI/helia-core-tester/commit/4b82314933af8aa38773c327b143c6ea27bc3dcc))
* **cli:** keep --json stdout clean by routing build, flash and generation output to stderr ([e5301e9](https://github.com/AmbiqAI/helia-core-tester/commit/e5301e9e26fc5b9976c91d4a61451f110a3e941a))
* **cli:** preflight hardening for the hardware commands ([8dcb09d](https://github.com/AmbiqAI/helia-core-tester/commit/8dcb09d1e58c18fe24c3c2a399e67664c16aeb5f))
* **cli:** validate options before probe resolution and print one-line pipeline errors ([8fcc540](https://github.com/AmbiqAI/helia-core-tester/commit/8fcc5402ca78a4e3bd7b858fcadf8bafe44321a9))
* copilot code review fixes ([5358948](https://github.com/AmbiqAI/helia-core-tester/commit/5358948e1813b38ebe58f4ccbadb2df895df66da))
* correct scatter_nd output_strides computation ([ed26d3f](https://github.com/AmbiqAI/helia-core-tester/commit/ed26d3f8736cd5f1369b299dd2dbe4b89d46d4c6))
* **coverage:** bypass compiler caches for instrumented kernels ([#181](https://github.com/AmbiqAI/helia-core-tester/issues/181)) ([9febd25](https://github.com/AmbiqAI/helia-core-tester/commit/9febd25ed7252bb26aef99911c8e2b0d3f1742e2))
* **coverage:** reject missing required suite inputs ([#170](https://github.com/AmbiqAI/helia-core-tester/issues/170)) ([f2a7359](https://github.com/AmbiqAI/helia-core-tester/commit/f2a73596ced2c2dcdd0a67b8759408a92f497e34))
* **discovery:** stop CMSIS_NN_REPO_ROOT from falsely validating a kernel checkout ([#94](https://github.com/AmbiqAI/helia-core-tester/issues/94)) ([f584ce9](https://github.com/AmbiqAI/helia-core-tester/commit/f584ce9fdb21318586af82174882ee7b8ed019fb))
* **firmware:** wrap-proof cursor capacity check; fail closed on a missing memory region ([b6c745b](https://github.com/AmbiqAI/helia-core-tester/commit/b6c745bf2aab28d0573ca96fce4ef46e970a7fab))
* **fvp:** carry the generation manifest as an active test list into build/run ([#86](https://github.com/AmbiqAI/helia-core-tester/issues/86)) ([93c6dde](https://github.com/AmbiqAI/helia-core-tester/commit/93c6ddee6efcd29ade1cb3eb302001270f41ab2a))
* **generation:** feed the hoisted bias of quantized dilated conv and depthwise cases at accumulator scale ([#114](https://github.com/AmbiqAI/helia-core-tester/issues/114)) ([920a748](https://github.com/AmbiqAI/helia-core-tester/commit/920a7489e3da392419c8432767fc0be9dd08ab3a))
* **generation:** include rolling-buffer size in transpose-conv scratch bound ([e8c5215](https://github.com/AmbiqAI/helia-core-tester/commit/e8c521510627b1df82ee4343e3f9fab4182b254b))
* **generation:** LSTM nested-layout fallback is parents[6], not [5] ([8762129](https://github.com/AmbiqAI/helia-core-tester/commit/8762129c397df28eaf3e909998b5649a78ff1da2))
* **generation:** LSTM schema path stays lenient without a checkout ([a6018c6](https://github.com/AmbiqAI/helia-core-tester/commit/a6018c6f68f83749803c3eb5b0333bfebc775a14))
* **generation:** mirror the kernel's float32 resize nearest-neighbour index ([#130](https://github.com/AmbiqAI/helia-core-tester/issues/130)) ([4c07043](https://github.com/AmbiqAI/helia-core-tester/commit/4c070431b666795fa2ca1f3a780ab65b5818e971))
* **generation:** preserve declared batches in model conversion ([#162](https://github.com/AmbiqAI/helia-core-tester/issues/162)) ([c4268a2](https://github.com/AmbiqAI/helia-core-tester/commit/c4268a29bb1b7888d18caec8cae814214dc29e5f))
* **generation:** require_cmsis_nn_root checks Source/ too ([2eb71c4](https://github.com/AmbiqAI/helia-core-tester/commit/2eb71c41fc7ded4f69032ca6857580cfddd5e661))
* **generation:** resolve the ns-cmsis-nn checkout correctly on a standalone clone ([255ee94](https://github.com/AmbiqAI/helia-core-tester/commit/255ee9464bcda1dca20ce61a95da495f86948874))
* **generation:** route the two parents[N] checkout guesses through CMSIS_NN_ROOT ([2847108](https://github.com/AmbiqAI/helia-core-tester/commit/28471087758ad6cd53f7389128d9567a0abc39a6))
* **generation:** serialize expected float arrays at full precision ([#84](https://github.com/AmbiqAI/helia-core-tester/issues/84)) ([3e229c7](https://github.com/AmbiqAI/helia-core-tester/commit/3e229c77ba3c578a08e85939bf9f3decc5395e9e))
* **generation:** stamp the sigmoid table and LSTM schema as checkout inputs ([15e83ce](https://github.com/AmbiqAI/helia-core-tester/commit/15e83ce36a71aabe73a1a3b756fb62318d0b53ea))
* **gru:** render the fault template's kernel name from context ([6f8d97e](https://github.com/AmbiqAI/helia-core-tester/commit/6f8d97e0cbc234e0c999291c5a30d22beab23b2d))
* **hardware:** fail when every shared case is flagged ([fc0c075](https://github.com/AmbiqAI/helia-core-tester/commit/fc0c0750900458462f9a46996cfeff7642ffce63))
* **hardware:** forward frozen through configure and build ([82cc194](https://github.com/AmbiqAI/helia-core-tester/commit/82cc1946ccaef0b4a45d05df3dba4e80844e3666))
* **hardware:** gate non-finite counters and validate limits ([ddc0b57](https://github.com/AmbiqAI/helia-core-tester/commit/ddc0b57603dae21b4e37c35db5515bd952ade540))
* **hardware:** preserve float classification and generated masks ([#180](https://github.com/AmbiqAI/helia-core-tester/issues/180)) ([95849f3](https://github.com/AmbiqAI/helia-core-tester/commit/95849f3fc587f3f0d959e26727ce8e34475d4973)), closes [#139](https://github.com/AmbiqAI/helia-core-tester/issues/139)
* **hardware:** refuse a kernel checkout that overlaps the app ([a70f5f8](https://github.com/AmbiqAI/helia-core-tester/commit/a70f5f8078cf8ada0612a716a24bf4be0c84afbf))
* **hardware:** seed modules.cmake once, resolve local kernels ([b2f3c41](https://github.com/AmbiqAI/helia-core-tester/commit/b2f3c4197c92f2003498653e8d17abed80bdc5f2))
* **perf-stream:** apply --precision when generating, not only when discovering ([7c6ef28](https://github.com/AmbiqAI/helia-core-tester/commit/7c6ef281a1cfe805cfb5a7fe95f2dfea2c78b357))
* **perf-stream:** board-key the size probe and build it with the toolchain on PATH ([9008aea](https://github.com/AmbiqAI/helia-core-tester/commit/9008aea42fc4ba3fea12f1a2a86b8a3aa608fd27))
* **perf-stream:** bound counters per pass regardless of chaining and wrap plan-size errors ([c20f114](https://github.com/AmbiqAI/helia-core-tester/commit/c20f11489595188f4ca2ed93e01aa9d88ce3d0a9))
* **perf-stream:** bound every outbound frame by max_rx_payload and thread project_root into the ELF probes ([cefe297](https://github.com/AmbiqAI/helia-core-tester/commit/cefe297a9be982ffabf648e9ad4c8a1a8c681f30))
* **perf-stream:** cap case ids at 95 bytes to match firmware cursor_text() ([4733be9](https://github.com/AmbiqAI/helia-core-tester/commit/4733be9e3630d5499c566e8a738006aa31b92e42))
* **perf-stream:** derive bundle counter columns from the selected passes ([3fcbc11](https://github.com/AmbiqAI/helia-core-tester/commit/3fcbc11f5f56449a27ea9f774e45701d10bbe9ec))
* **perf-stream:** fetch CMSIS_5 lazily before the first hardware configure ([7f1bcd9](https://github.com/AmbiqAI/helia-core-tester/commit/7f1bcd98da21173d813f789fe94ede3dc15b4f9a))
* **perf-stream:** host-side path and toolchain helpers for the hardware CLI ([d5c0c4f](https://github.com/AmbiqAI/helia-core-tester/commit/d5c0c4f996d3c5c51a170eff3c23583c1d66cf0e))
* **perf-stream:** include capability_flags in the cross-batch consistency check ([cea1cdf](https://github.com/AmbiqAI/helia-core-tester/commit/cea1cdfd7270fa9f8ba4723b0cd4e977579b23e0))
* **perf-stream:** key the flash skip on a per-build firmware id the board confirms ([405215d](https://github.com/AmbiqAI/helia-core-tester/commit/405215d17f91cb95a554166babaea2cd607d3afc))
* **perf-stream:** last write_text(newline=) call, fake BLOB_CHUNK rx bound, positive max_passes ([58e5a19](https://github.com/AmbiqAI/helia-core-tester/commit/58e5a19385523d3645b1ba8b71408265392abac2))
* **perf-stream:** number the emit harness's catalog pages with monotonic sequence ids ([8aee601](https://github.com/AmbiqAI/helia-core-tester/commit/8aee6015df6ddb33becb783852636e8e8411c4a0))
* **perf-stream:** pass the board's workspace_bytes to the firmware configure ([e050366](https://github.com/AmbiqAI/helia-core-tester/commit/e0503663edee3a889b2fa5bef5577446ea803177))
* **perf-stream:** preserve zip contracts on older Python ([#168](https://github.com/AmbiqAI/helia-core-tester/issues/168)) ([ae39f92](https://github.com/AmbiqAI/helia-core-tester/commit/ae39f920f133561d59576429340d5663fd8d20bb))
* **perf-stream:** re-enumerate probes once when the first scan finds none ([2f97ebd](https://github.com/AmbiqAI/helia-core-tester/commit/2f97ebd102a370f4cc11419cc24dadd83cde1362))
* **perf-stream:** refuse duplicate case ids before any plan is sent ([d6f1343](https://github.com/AmbiqAI/helia-core-tester/commit/d6f1343c503e12b227512a841470cfb3d0f499d5))
* **perf-stream:** refuse more than 16 PMU passes per plan on the host ([9973acb](https://github.com/AmbiqAI/helia-core-tester/commit/9973acb382c2e04aa277d6f60d08425d4b56afb6))
* **perf-stream:** refuse non-positive session limits from TARGET_INFO ([5566dc5](https://github.com/AmbiqAI/helia-core-tester/commit/5566dc5b8cb874f14e84c0264750d0f0d5300922))
* **perf-stream:** reject malformed payloads and mixed-firmware batches ([64055b1](https://github.com/AmbiqAI/helia-core-tester/commit/64055b17dfe64e2a0007333592ac45c5f90aeaaf))
* **perf-stream:** resolve JLinkExe for the flash target the way the library is resolved ([3b5b200](https://github.com/AmbiqAI/helia-core-tester/commit/3b5b200d3c62bd22b1bcab1b7584146f536b0cdd))
* **perf-stream:** resolve the J-Link library the way the lab runners expect (HPX_JLINK_DLL, JLINK_PATH, JLinkExe on PATH) ([01085fc](https://github.com/AmbiqAI/helia-core-tester/commit/01085fc0b5938f40f612eda453bd904df8a48927))
* **perf-stream:** seed bundle counter columns by event id and tighten the fake target ([5a67442](https://github.com/AmbiqAI/helia-core-tester/commit/5a67442af6a15d99c4b2032bc15d0e3c047aef11))
* **perf-stream:** stamp the build id into the ELF again ([7ed8e11](https://github.com/AmbiqAI/helia-core-tester/commit/7ed8e118d1e65c8dc87fb30a0fb587b018062d35))
* **perf-stream:** stop passing newline= to Path.write_text (Python 3.8/3.9) ([6c55f96](https://github.com/AmbiqAI/helia-core-tester/commit/6c55f96400301b74d256a38f9bee7bfdcf5fc403))
* **prelu:** fix S16 PReLUScalar validation and dispatcher coverage gaps ([76fbd84](https://github.com/AmbiqAI/helia-core-tester/commit/76fbd849eb21e4222d305c477a2288570c1022b0))
* **reduce:** preserve integer case sensitivity ([065af2c](https://github.com/AmbiqAI/helia-core-tester/commit/065af2ce9f2798b640203e43baa45da757b0fabe)), closes [#155](https://github.com/AmbiqAI/helia-core-tester/issues/155)
* remove is_variable and .tobytes() from TensorSpec usage ([f7649bf](https://github.com/AmbiqAI/helia-core-tester/commit/f7649bf16924a1bac763c67418c01864669c4d76))
* **review:** harden bias extraction, close invariant gaps, right-size tolerances ([621fb4d](https://github.com/AmbiqAI/helia-core-tester/commit/621fb4d084daefaa73b3dd320acfcab91dc7c1ed))
* **runtime:** classify non-finite float operands before the tolerance ([#96](https://github.com/AmbiqAI/helia-core-tester/issues/96)) ([cfe0be7](https://github.com/AmbiqAI/helia-core-tester/commit/cfe0be7ea33953c6163e0d8d347df2812fe18cf0))
* sample distinct operands for Add/Mul/MinMax goldens ([52bdff9](https://github.com/AmbiqAI/helia-core-tester/commit/52bdff9f9364686b7ef80e178ad90bc62292a27b)), closes [#48](https://github.com/AmbiqAI/helia-core-tester/issues/48)
* sample distinct operands in the float sub golden path ([0a7e810](https://github.com/AmbiqAI/helia-core-tester/commit/0a7e8103984e1b280fbd4ccb684b9286e2a926d4))
* sub row scalar channel broadcast regression ([d0f5384](https://github.com/AmbiqAI/helia-core-tester/commit/d0f53849659ceb635a25095993fd6a46366c7a85))
* **tolerance:** restore tanh f16 overrides -- they were load-bearing for ([b0e9230](https://github.com/AmbiqAI/helia-core-tester/commit/b0e92308c92e9bba7c33b7d6d6f2df9463dffd43))
* **validation:** float outputs always get float comparison ([#54](https://github.com/AmbiqAI/helia-core-tester/issues/54)) ([d56eae7](https://github.com/AmbiqAI/helia-core-tester/commit/d56eae7d371926ee10727c4de346e1b6a7fc0ce7))


### Performance

* **perf-stream:** bridge generated cases once per hardware stream ([66d9718](https://github.com/AmbiqAI/helia-core-tester/commit/66d971834beb084e3213c5c4a7f2f4e71dfdea6a))


### Refactoring

* eliminate numpy fallback, use TFLite interpreter for all reference data ([240c34d](https://github.com/AmbiqAI/helia-core-tester/commit/240c34df5e6102d405aabc1d5f22fdce20d941de))
* **firmware:** generate the adapter bodies into benchmark_server_adapters.gen.c ([1cbbea7](https://github.com/AmbiqAI/helia-core-tester/commit/1cbbea70a55412ad1c8ccbbf149fa2720fec480b))
* **firmware:** render the complete kernel dispatch into the generated file ([4bfba5b](https://github.com/AmbiqAI/helia-core-tester/commit/4bfba5b9cff39295b42ab4d00562333a9ba88264))
* **hardware:** consolidate generated bundle assembly ([#185](https://github.com/AmbiqAI/helia-core-tester/issues/185)) ([3c6b0b7](https://github.com/AmbiqAI/helia-core-tester/commit/3c6b0b7ce4c59c8d603500822d1864a2c81bd757)), closes [#184](https://github.com/AmbiqAI/helia-core-tester/issues/184)
* **hardware:** name the layout constants ([9330e2c](https://github.com/AmbiqAI/helia-core-tester/commit/9330e2c5e953ecb0283397a2ac5863026666c535))
* **hardware:** reuse typed extraction in unary builders ([#183](https://github.com/AmbiqAI/helia-core-tester/issues/183)) ([1314418](https://github.com/AmbiqAI/helia-core-tester/commit/131441891dec942a3054e69708c78467835e2b1c))
* **hardware:** route build output paths through firmware_build ([31b8dcd](https://github.com/AmbiqAI/helia-core-tester/commit/31b8dcd75cb486904d53e9fd7fd0b69c7a8010f6))
* **hardware:** slim the NSX facade after review ([2496e77](https://github.com/AmbiqAI/helia-core-tester/commit/2496e7766f7cc216743c6ec3d65b063a10925843))
* **perf-stream:** board-generic session runner batched from TARGET_INFO ([3aee1c3](https://github.com/AmbiqAI/helia-core-tester/commit/3aee1c334379c4d033df683cde4685574b807d20))
* **perf-stream:** default the memory report to the board-keyed build dir ([9818a0e](https://github.com/AmbiqAI/helia-core-tester/commit/9818a0eea61e589617ad87dc3839cb0d2010d87e))
* **perf-stream:** HCTP v3 vocabulary, drop unused messages, advertise session limits ([d0487c5](https://github.com/AmbiqAI/helia-core-tester/commit/d0487c5789fc20558b6158128affad0e8ea75911))
* **perf-stream:** HCTP v3 vocabulary, one codec, board-generic runner, generated adapters file ([4f10de6](https://github.com/AmbiqAI/helia-core-tester/commit/4f10de6aa77c43b5cd711d9eab8807abb5d3fb95))
* **perf-stream:** move build/flash, summaries and orchestration out of the CLI ([069b6dd](https://github.com/AmbiqAI/helia-core-tester/commit/069b6dd5a2442fe5fbf455e344ab64b72409185c))
* **perf-stream:** one board-keyed memory report ([f070fcb](https://github.com/AmbiqAI/helia-core-tester/commit/f070fcbb4765b62c8c843501995a7e9c2faea116))
* **perf-stream:** one Python codec for every HCTP payload ([3f5ff21](https://github.com/AmbiqAI/helia-core-tester/commit/3f5ff21dce46618581507b49d5b7562bccb5cf1d))
* rename perf_stream package and firmware dir to hardware ([4905614](https://github.com/AmbiqAI/helia-core-tester/commit/4905614c5ced2e1e25447f69c9dd36e66dcf1a70))
* **transpose_conv:** compute rolling-buffer size once, shared by both branches ([7a9e90b](https://github.com/AmbiqAI/helia-core-tester/commit/7a9e90b6a93fcd64ced407f40102e2f3cac6eb54))
* use TFLite interpreter for reference data generation ([7219d26](https://github.com/AmbiqAI/helia-core-tester/commit/7219d262f1f7d048edc998e0366244e8b44a825d))


### Docs

* **abs:** update stale comment to the renamed float symbols ([5fdff04](https://github.com/AmbiqAI/helia-core-tester/commit/5fdff0446c2b435308b8b1e832ca5f2a3b42cd0f))
* **audit:** mutation-test remaining float op families (issue [#53](https://github.com/AmbiqAI/helia-core-tester/issues/53) item 2) ([#80](https://github.com/AmbiqAI/helia-core-tester/issues/80)) ([6caf2b1](https://github.com/AmbiqAI/helia-core-tester/commit/6caf2b1a1760646ebf82e7164607242dd8246684))
* **audit:** re-measure float tolerance headroom post-[#54](https://github.com/AmbiqAI/helia-core-tester/issues/54) (issue [#53](https://github.com/AmbiqAI/helia-core-tester/issues/53) item 1) ([#79](https://github.com/AmbiqAI/helia-core-tester/issues/79)) ([6de6fb2](https://github.com/AmbiqAI/helia-core-tester/commit/6de6fb22d333ed2986b48238a8101b827356ae9e))
* describe HCTP v2 PMU passes, --pmu-counters and the hardware proof ([76d683e](https://github.com/AmbiqAI/helia-core-tester/commit/76d683e5dc6bffc6a9feafb7f4460fe734be6380))
* describe the hardware CLI and update command examples ([48286f1](https://github.com/AmbiqAI/helia-core-tester/commit/48286f1db3b5874a688650d9f0c1428037a818a4))
* **hardware:** say NSX supplies cmake/nsx on sync ([11c6202](https://github.com/AmbiqAI/helia-core-tester/commit/11c62026b2aea6ca9ebc2794f1cf6c31977c4457))
* HCTP v3 message table, TARGET_INFO limits, codec and host modules ([d8d29d7](https://github.com/AmbiqAI/helia-core-tester/commit/d8d29d7ae3ddd5e827cf75638cb88e9dfb19b9a6))
* **perf-stream:** review follow-ups after the rebase onto main ([af3a655](https://github.com/AmbiqAI/helia-core-tester/commit/af3a6554e1f95f6bce3a1e577c8cd8487f64ea01))
* **pmu:** drop the duplicated word in the unaligned-access description ([3ea22d2](https://github.com/AmbiqAI/helia-core-tester/commit/3ea22d2627f417c3207ab19ed8dda87f5aee0b21))

## [0.2.0](https://github.com/AmbiqAI/helia-core-tester/compare/v0.1.0...v0.2.0) (2026-04-02)


### Features

* add tag release workflow and update release process documentation ([dc5ce4c](https://github.com/AmbiqAI/helia-core-tester/commit/dc5ce4c2b405c6dd3df6712d92a08790960d3c9a))


### Bug Fixes

* Correct sqrt LUT to run without numpy errors ([e67d504](https://github.com/AmbiqAI/helia-core-tester/commit/e67d5048414b2bd8a45809ef45456801986bde50))
* document to trigger first release ([d80abc2](https://github.com/AmbiqAI/helia-core-tester/commit/d80abc20687ab5df4f920f944e5e717110d87cd3))

## Changelog

All notable changes to this project will be documented in this file.

This file is managed by release-please.

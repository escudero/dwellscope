# Sample clip

🎬 **restricted_room:** [watch on YouTube](https://youtu.be/CjTEk33Sr8Y)

A 45 s clip (1890 × 1080, 24 fps) from a camera inside a restricted room, looking at its door. Three people go in and out. It is the sample for [`scripts/dwellscope.py`](../scripts/dwellscope.py) (see [Trying your own video](../README.md#trying-your-own-video)).

The video is not in the repository. Download it into this folder, from the repository root:

```bash
uv pip install yt-dlp
yt-dlp -f "bv*[ext=mp4]" -o samples/restricted_room.mp4 "https://youtu.be/CjTEk33Sr8Y"
```

The door line in [`restricted_room.door.json`](restricted_room.door.json) is in pixels of the 1890 × 1080 original. If the download comes at another size, check the line with `--check` and scale the points. YouTube re-encodes the video, so the times can differ from the README by a fraction of a second.

---

# Vídeo de exemplo

🎬 **restricted_room:** [assista no YouTube](https://youtu.be/CjTEk33Sr8Y)

Um vídeo de 45 s (1890 × 1080, 24 fps) de uma câmera dentro de uma sala restrita, olhando para a porta. Três pessoas entram e saem. É o exemplo do [`scripts/dwellscope.py`](../scripts/dwellscope.py) (veja [Testando no seu próprio vídeo](../README.pt-BR.md#testando-no-seu-próprio-vídeo)).

O vídeo não está no repositório. Baixe-o para esta pasta, a partir da raiz do repositório:

```bash
uv pip install yt-dlp
yt-dlp -f "bv*[ext=mp4]" -o samples/restricted_room.mp4 "https://youtu.be/CjTEk33Sr8Y"
```

A linha da porta em [`restricted_room.door.json`](restricted_room.door.json) está em pixels do original de 1890 × 1080. Se o download vier em outro tamanho, confira a linha com `--check` e ajuste os pontos na mesma proporção. O YouTube recodifica o vídeo, então os tempos podem variar uma fração de segundo em relação ao README.

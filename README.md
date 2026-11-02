# vasp-metal-backend

> [!IMPORTANT]
> Independent, unofficial and experimental. This project is not affiliated with
> or endorsed by the VASP Software GmbH. It contains no VASP source code, VASP
> executables, proprietary patches or PAW/POTCAR datasets. Bring your own
> separately licensed VASP and potentials.

An opt-in Apple Metal backend for one numerical operation used by VASP 6.6.1
`vasp_std` on Apple Silicon. The backend is attached through a documented public
interface at link time; licensed VASP objects are relinked privately and never
leave the local machine.

## Status

Experimental (v0.1 in progress). Default off. Correctness and dispatch are the
goal; no performance claim is made.

## License

The original work in this repository is released under the MIT License (see
`LICENSE`). It grants no rights to VASP software, source, executables or
PAW/POTCAR datasets.

if __package__:
    from .pricing_catalog import main
else:
    from pricing_catalog import main


if __name__ == "__main__":
    raise SystemExit(main())

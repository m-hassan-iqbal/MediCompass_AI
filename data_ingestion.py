def sync_google_drive_public_folders(
    folder_urls: list[str], target_dir: str
) -> list[str]:
    """
    Downloads documents from public Google Drive folder links or direct file links using gdown.
    Uses regex ID extraction to handle all sharing URL formats (including /u/0/, ?usp=sharing, etc).
    Compatible with gdown 6.2+.
    """
    downloaded_files = []
    if not folder_urls:
        return downloaded_files

    try:
        import gdown  # Lazy import
    except ImportError:
        logger.warning("gdown library not installed. Cannot sync Google Drive folders.")
        return downloaded_files

    os.makedirs(target_dir, exist_ok=True)

    for url in folder_urls:
        url = url.strip()
        if not url or url.startswith("#"):
            continue

        try:
            logger.info(f"Syncing Google Drive source: {url}")
            folder_match = re.search(r"folders/([a-zA-Z0-9_-]+)", url)
            file_match = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)

            if folder_match:
                # Folder download using extracted ID
                folder_id = folder_match.group(1)
                try:
                    res = gdown.download_folder(
                        id=folder_id,
                        output=target_dir,
                        quiet=False,
                        use_cookies=False,
                    )
                except Exception as fe1:
                    logger.warning(f"Failed download_folder by id ({fe1}). Trying by URL.")
                    res = gdown.download_folder(
                        url=url,
                        output=target_dir,
                        quiet=False,
                        use_cookies=False,
                    )
                if res:
                    downloaded_files.extend([str(r) for r in res])
            elif file_match:
                # Direct file download using extracted ID
                file_id = file_match.group(1)
                try:
                    res = gdown.download(
                        id=file_id,
                        output=os.path.join(target_dir, ""),
                        quiet=False,
                    )
                except Exception as dfe1:
                    logger.warning(f"Failed download by id ({dfe1}). Trying by URL.")
                    res = gdown.download(
                        url=url,
                        output=os.path.join(target_dir, ""),
                        quiet=False,
                    )
                if res:
                    downloaded_files.append(str(res))
            else:
                # Fallback to direct URL
                res = gdown.download(
                    url=url,
                    output=os.path.join(target_dir, ""),
                    quiet=False,
                )
                if res:
                    downloaded_files.append(str(res))
        except Exception as e:
            logger.error(f"Failed to sync Google Drive link {url}: {e}")

    return downloaded_files

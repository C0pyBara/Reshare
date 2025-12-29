from telethon import TelegramClient

from config import API_ID, API_HASH, SESSION_NAME


async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()

    async for dialog in client.iter_dialogs():
        ent = dialog.entity
        print(
            "title/username:",
            getattr(ent, "title", getattr(ent, "username", None)),
            "id:",
            getattr(ent, "id", None),
        )

    await client.disconnect()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())



from setuptools import setup, find_packages

setup(
    name="LangTorch",
    version="1.0.6",
    description="Framework for intuitive LLM application development with tensors.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Adam Sobieszek",
    author_email="contact@langtorch.org",
    url="https://github.com/AdamSobieszek/LangTorch",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch >= 2.1.0",
        "aiohttp",
        "nest_asyncio",
        "openai>=1.2.4",
        "tiktoken",
        "pandas",
        "retry",
        "pyparsing",
        "pypandoc",
        "transformers",
        'omegaconf',
        'hydra-core',
        "markdown-it-py"
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="LangTorch, PyTorch, LLM, Language Models, Chat, Chains",
    package_data={
        'langtorch': ['conf/defaults.yaml', 'conf/new_session_template.yaml','conf/overrides.yaml', 'methods/prompts.yaml'],
    },
    include_package_data=True,
)

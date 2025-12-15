package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	if err := Build(ctx); err != nil {
        fmt.Println("Error:", err)
        panic(err)
    }
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	// Mirror the root of our repository
	itu_sdse_project := client.Host().Directory(".")

	// Before running any py files, install requirements
	require := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("/repo", itu_sdse_project).
		WithWorkdir("/repo/src").
		WithExec([]string{"python", "--version"})

	require = require.WithExec([]string{
		"bash", "-lc",
		"pip install --upgrade pip",
	})
	
	require = require.WithExec([]string{
		"bash", "-lc",
		"python -m pip install -r /repo/requirements.txt",
	})
	_, err = require.Stdout(ctx)
	if err != nil {
		return err
	}
	
	fmt.Println("requirements installed")

	config = require.WithExec([]string{"python", "config.py"})
	_, err = config.Stdout(ctx)
	if err != nil {
		return err
	}

	fmt.Println("config.py ran")

	data = config.WithExec([]string{"python", "dataset.py"})
	_, err = data.Stdout(ctx)
	if err != nil {
		return err
	}

	fmt.Println("dataset.py ran")

	features = data.WithExec([]string{"python", "features.py"})
	_, err = features.Stdout(ctx)
	if err != nil {
		return err
	}

	fmt.Println("features.py ran")

	train = features.WithExec([]string{"python", "modeling/train.py"})
	_, err = train.Stdout(ctx)
	if err != nil {
		return err
	}

	fmt.Println("train.py ran")

	selection = train.WithExec([]string{"python", "modeling/model_selection.py"})
	_, err = selection.Stdout(ctx)
	if err != nil {
		return err
	}

	fmt.Println("model_selection.py ran")

	_, err = selection.
		Directory("/repo/artifacts").
		Export(ctx, "artifacts")
	if err != nil {
		return err
	}

	_, err = selection.
		Directory("/repo/data").
		Export(ctx, "data")
	if err != nil {
		return err
	}

	_, err = selection.
		Directory("/repo/mlruns").
		Export(ctx, "mlruns")
	if err != nil {
		return err
	}
	return nil
}
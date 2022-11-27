package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import lombok.Data;

import java.util.List;

import static com.deep.framework.ast.lexer.TokenType.THROW;

@Data
public class ThrowStatement extends Statement {
    private Expression expression;

    public ThrowStatement(Node prarent){
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof ThrowStatement) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (a.equals(THROW)) {
                ThrowStatement statement = new ThrowStatement(node);
                statement.getChildrens().addAll(node.getChildrens());
                statement.getChildrens().remove(a);
                node.replace(a, statement);
                node.getChildrens().removeAll(statement.getChildrens());
            }
        });
    }
}
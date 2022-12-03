package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import static com.deep.framework.ast.lexer.TokenType.SWITCH;

@Data
public class SwitchStatement extends Statement {
    private Expression expression;
    private BlockStatement body;

    public SwitchStatement(Node prarent, Expression expression, BlockStatement body) {
        super(prarent);
        this.expression = expression;
        this.body = body;

        this.expression.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(expression, body);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((c, m, n) -> {
                if (m.equals(SWITCH) && n instanceof ParametersExpression) {
                    //create SwitchNode and set Prarent，Parameters
                    SwitchStatement statement = new SwitchStatement(node, (Expression) n, (BlockStatement) b);
                    SwitchEntry.parser(b);
                    //remove SwitchNode and Parameters
                    node.replace(a, statement);
                    node.getChildrens().remove(b);
                    list.remove(b);
                }
            });
        });
    }

}